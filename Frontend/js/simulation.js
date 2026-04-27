/**
 * simulation.js – Digital Twin Simulation Engine Frontend Logic (v5.0)
 * ====================================================================
 * Handles slider interactions, API calls, chart rendering, and
 * scenario comparison for the Tax Policy Simulation page.
 *
 * Charts (all Canvas-based):
 *   1. Revenue Projection Line Chart (fixed legend overflow)
 *   2. Risk Distribution Donut
 *   3. Waterfall – Parameter Contributions
 *   4. Tornado – Sensitivity Analysis
 *   5. Radar – Scenario Comparison
 *   6. Bubble – Industry Map
 *   7. Gauge – Overall Risk Meter
 *   8. Heatmap – Industry × Risk Level
 *   9. Slope – Industry Rank Change
 *  10. Stacked Area – Revenue Breakdown
 */

const SIM_API = `${API_BASE}/simulation`;

// ── State ──────────────────────────────────────────────────
let currentScenario = null;
let baselineData = null;
let sensitivityData = null;
let contributionsData = null;
let heatmapData = null;
let historicalData = null;
let hypothesisPayload = null;
let externalSignalPayload = null;
let activeHypothesisTab = 1;
let simulationRequestToken = 0;
let advancedRequestToken = 0;
const charts = Object.create(null);

// ── Color Palette ──────────────────────────────────────────
const SIM_COLORS = {
    primary: "#002147",
    sky: "#0284c7",
    violet: "#7c3aed",
    emerald: "#10b981",
    amber: "#f59e0b",
    orange: "#f97316",
    rose: "#ef4444",
    slate: "#64748b",
    slateLight: "#94a3b8",
    slateBg: "#f8fafc",
    gridLine: "#e2e8f0",
    white: "#ffffff",
};

const RISK_COLORS = { low: "#10b981", medium: "#f59e0b", high: "#f97316", critical: "#ef4444" };
const RISK_LABELS = { low: "Thấp", medium: "Trung bình", high: "Cao", critical: "Nghiêm trọng" };

function hasChartJs() {
    return typeof Chart !== "undefined";
}

function drawCanvasEmptyState(canvas, message, height = 260) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement?.getBoundingClientRect();
    const width = Math.max(220, rect?.width || 320);
    const h = Math.max(180, height);
    canvas.width = width * dpr;
    canvas.height = h * dpr;
    canvas.style.width = width + "px";
    canvas.style.height = h + "px";
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, h);
    ctx.fillStyle = SIM_COLORS.slateBg;
    ctx.fillRect(0, 0, width, h);
    ctx.fillStyle = SIM_COLORS.slate;
    ctx.font = "12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(message, width / 2, h / 2);
}

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
    return function (...args) {
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
        "sim-cpi-rate": "cpi_pct",
        "sim-unemployment": "unemployment_pct",
        "sim-exchange-delta": "exchange_rate_delta_pct",
        "sim-projection-years": "projection_years"
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
        const [baseRes, histRes] = await Promise.all([
            secureFetch(`${SIM_API}/baseline`),
            secureFetch(`${SIM_API}/historical-trends`)
        ]);
        if (baseRes.ok) {
            baselineData = await baseRes.json();
            renderKPIs(baselineData, null);
        }
        if (histRes.ok) {
            historicalData = await histRes.json();
            renderHistoricalChart(historicalData, currentScenario);
        }
        await loadHypotheses();
    } catch (e) {
        console.error("[Simulation] Failed to load baseline/historical:", e);
    }
}

async function loadHypotheses(refresh = true) {
    try {
        const [hypRes, signalRes] = await Promise.all([
            secureFetch(`${SIM_API}/hypotheses?refresh=${refresh ? "true" : "false"}`),
            secureFetch(`${SIM_API}/external-signals/snapshot?limit=8`),
        ]);
        if (!hypRes.ok) return;
        hypothesisPayload = await hypRes.json();
        externalSignalPayload = signalRes.ok ? await signalRes.json() : null;
        renderHypothesisPanel(hypothesisPayload, externalSignalPayload);
    } catch (err) {
        console.error("[Simulation] Failed to load hypotheses:", err);
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
        cpi_pct: parseFloat(document.getElementById("sim-cpi-rate")?.value || 3.5),
        unemployment_pct: parseFloat(document.getElementById("sim-unemployment")?.value || 2.3),
        exchange_rate_delta_pct: parseFloat(document.getElementById("sim-exchange-delta")?.value || 0),
        projection_years: parseInt(document.getElementById("sim-projection-years")?.value || 5, 10),
    };
}

// ── Run Simulation ─────────────────────────────────────────
async function runSimulation(name) {
    const params = gatherParams();
    const requestToken = ++simulationRequestToken;
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
        if (requestToken !== simulationRequestToken) return;
        currentScenario = await res.json();
        renderKPIs(baselineData, currentScenario);
        renderChart(currentScenario);
        renderIndustryTable(currentScenario);
        renderRiskDist(currentScenario);
        renderGauge(currentScenario);
        renderBubbleChart(currentScenario);
        renderSlopeChart(currentScenario);
        renderRadarChart(currentScenario, params);
        renderProjectionInsights(currentScenario);
        if (!hypothesisPayload) await loadHypotheses(false);

        // Render composite health score specifically
        renderHealthGauge(currentScenario.scenario_health_score);
        renderStackedAreaChart(currentScenario);
        if (historicalData) renderHistoricalChart(historicalData, currentScenario);

        // Load advanced chart data in parallel
        loadAdvancedChartData(params, requestToken);
    } catch (e) {
        console.error("[Simulation] Error:", e);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<span class="material-symbols-outlined text-[16px]">play_arrow</span> Chạy Mô phỏng';
        }
    }
}

async function loadAdvancedChartData(params, requestToken) {
    const advancedToken = ++advancedRequestToken;
    const mcCanvas = document.getElementById("sim-montecarlo-chart");
    try {
        if (mcCanvas) mcCanvas.style.opacity = "0.5";
        const [contribRes, sensitivityRes, heatmapRes, monteCarloRes] = await Promise.all([
            secureFetch(`${SIM_API}/parameter-contributions`, { method: "POST", body: JSON.stringify(params) }),
            secureFetch(`${SIM_API}/sensitivity`, { method: "POST", body: JSON.stringify(params) }),
            secureFetch(`${SIM_API}/industry-risk-matrix`, { method: "POST", body: JSON.stringify(params) }),
            secureFetch(`${SIM_API}/monte-carlo`, { method: "POST", body: JSON.stringify(params) })
        ]);
        if (requestToken !== simulationRequestToken || advancedToken !== advancedRequestToken) return;

        if (contribRes.ok) {
            contributionsData = await contribRes.json();
            renderWaterfallChart(contributionsData);
        }
        if (sensitivityRes.ok) {
            sensitivityData = await sensitivityRes.json();
            renderTornadoChart(sensitivityData);
        }
        if (heatmapRes.ok) {
            heatmapData = await heatmapRes.json();
            renderHeatmap(heatmapData);
        }
        if (monteCarloRes.ok) {
            const mcData = await monteCarloRes.json();
            if (mcCanvas) mcCanvas.style.opacity = "1";
            renderMonteCarloChart(mcData);
        }
    } catch (e) {
        console.error("[Simulation] Advanced chart data error:", e);
    } finally {
        if (mcCanvas) mcCanvas.style.opacity = "1";
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

// ────────────────────────────────────────────────────────────
//  CHART 1: Revenue Projection Line Chart (BUG FIX: legend overflow)
// ────────────────────────────────────────────────────────────
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
    // BUG FIX: increased padding to prevent label overflow
    const pad = { top: 20, right: 30, bottom: 45, left: 80 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const data = scenario.quarterly_projection;
    const allVals = data.flatMap(d => [d.baseline_value, d.simulated_value]);
    const minV = Math.min(...allVals) * 0.95;
    const maxV = Math.max(...allVals) * 1.05;

    const xScale = (i) => pad.left + (i / (data.length - 1)) * plotW;
    const yScale = (v) => pad.top + plotH - ((v - minV) / (maxV - minV)) * plotH;

    // Background
    ctx.fillStyle = SIM_COLORS.slateBg;
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = SIM_COLORS.gridLine;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
        const y = pad.top + (i / 5) * plotH;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(W - pad.right, y);
        ctx.stroke();
        const val = maxV - (i / 5) * (maxV - minV);
        ctx.fillStyle = SIM_COLORS.slateLight;
        ctx.font = "10px Inter, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(val.toFixed(0) + " tỷ", pad.left - 10, y + 3);
    }

    const labelStride = data.length <= 8 ? 1 : data.length <= 20 ? 2 : 4;
    const shortQuarter = (q) => {
        const [quarter, year] = String(q).split("/");
        return `${quarter}/${String(year || "").slice(-2)}`;
    };

    // X labels (decimated for long horizons)
    ctx.textAlign = "center";
    data.forEach((d, i) => {
        if (i % labelStride !== 0 && i !== data.length - 1) return;
        ctx.fillStyle = SIM_COLORS.slateLight;
        ctx.font = "10px Inter, sans-serif";
        ctx.fillText(shortQuarter(d.quarter), xScale(i), H - pad.bottom + 18);
    });

    // Baseline line
    ctx.beginPath();
    ctx.strokeStyle = SIM_COLORS.slateLight;
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
    grad.addColorStop(0, SIM_COLORS.sky);
    grad.addColorStop(1, SIM_COLORS.violet);
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
        ctx.fillStyle = SIM_COLORS.sky;
        ctx.fill();
        ctx.strokeStyle = SIM_COLORS.white;
        ctx.lineWidth = 2;
        ctx.stroke();
    });

    // BUG FIX: Legend moved to HTML (see delinquency.html), no longer drawn on canvas
}

function renderProjectionInsights(scenario) {
    const container = document.getElementById("sim-projection-insights");
    if (!container || !scenario?.quarterly_projection?.length) return;
    const points = scenario.quarterly_projection;
    const first = points[0]?.simulated_value || 0;
    const last = points[points.length - 1]?.simulated_value || 0;
    const years = Math.max(1, Number(scenario?.parameters?.projection_years || 1));
    const cagr = first > 0 ? ((Math.pow(last / first, 1 / years) - 1) * 100) : 0;
    const yoyAvg = Number(scenario.avg_yoy_pct ?? NaN);
    const yoyMedian = Number(scenario.median_yoy_pct ?? NaN);
    const yoyDispersion = Number(scenario.yoy_dispersion_pct ?? NaN);
    const hasYoy = Number.isFinite(yoyAvg);
    const values = points.map((p) => p.simulated_value);
    const volatility = Math.max(...values) - Math.min(...values);
    const riskState = scenario.simulated_delinquency_rate >= 70 ? "Áp lực cao" : scenario.simulated_delinquency_rate >= 45 ? "Áp lực trung bình" : "Ổn định";
    const horizonHyp = (hypothesisPayload?.items || []).find((item) => Number(item.horizon_years) === years);
    const confidencePct = horizonHyp ? Math.round(Number(horizonHyp.confidence || 0) * 100) : Math.max(55, 82 - years * 3);

    container.innerHTML = `
        <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <p class="text-[10px] uppercase tracking-wider font-bold text-slate-500">CAGR</p>
            <p class="text-lg font-black ${cagr >= 0 ? "text-emerald-600" : "text-rose-600"}">${cagr >= 0 ? "+" : ""}${cagr.toFixed(2)}%</p>
        </div>
        <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <p class="text-[10px] uppercase tracking-wider font-bold text-slate-500">YoY trung bình</p>
            <p class="text-lg font-black ${hasYoy && yoyAvg >= 0 ? "text-emerald-600" : "text-rose-600"}">
                ${hasYoy ? `${yoyAvg >= 0 ? "+" : ""}${yoyAvg.toFixed(2)}%` : "Chưa đủ dữ liệu"}
            </p>
        </div>
        <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <p class="text-[10px] uppercase tracking-wider font-bold text-slate-500">Biến động dự phóng</p>
            <p class="text-lg font-black text-primary-container">${volatility.toFixed(1)} tỷ</p>
        </div>
        <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <p class="text-[10px] uppercase tracking-wider font-bold text-slate-500">Cảnh báo kịch bản</p>
            <p class="text-lg font-black ${scenario.simulated_delinquency_rate >= 70 ? "text-rose-600" : "text-amber-600"}">${riskState}</p>
        </div>
        <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <p class="text-[10px] uppercase tracking-wider font-bold text-slate-500">Độ tin cậy giả thuyết</p>
            <p class="text-lg font-black text-primary-container">${confidencePct}%</p>
        </div>
        <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <p class="text-[10px] uppercase tracking-wider font-bold text-slate-500">Số quý theo dõi</p>
            <p class="text-lg font-black text-primary-container">${points.length} quý</p>
        </div>
    `;

    const insightDetails = {
        growth: hasYoy
            ? `Tăng trưởng đang phản ánh quỹ đạo ${yoyAvg >= 0 ? "mở rộng" : "chậm lại"} với YoY trung bình ${yoyAvg.toFixed(2)}%, trung vị ${Number.isFinite(yoyMedian) ? yoyMedian.toFixed(2) : "N/A"}% và độ phân tán ${Number.isFinite(yoyDispersion) ? yoyDispersion.toFixed(2) : "N/A"}pp.`
            : "Chưa đủ chuỗi dữ liệu để tính YoY đáng tin cậy; hệ thống sẽ dùng thêm các chỉ số CAGR/volatility để đánh giá xu hướng.",
        risk: `Áp lực rủi ro hiện ở mức "${riskState}", cần theo dõi sát các ngành nhạy cảm trong 2 quý tới.`,
        volatility: `Độ dao động dự báo khoảng ${volatility.toFixed(1)} tỷ, cho thấy mức biến thiên ${volatility > 2500 ? "cao" : "vừa phải"}.`,
        confidence: `Độ tin cậy mô hình giả thuyết hiện ${confidencePct}%, được suy ra từ dữ liệu nội bộ + tín hiệu external.`,
    };
    const detailBox = document.getElementById("sim-insight-detail");
    if (detailBox) {
        detailBox.textContent = insightDetails.growth;
    }
    document.querySelectorAll("[data-sim-insight-chip]").forEach((chip) => {
        chip.onclick = () => {
            const key = chip.getAttribute("data-sim-insight-chip");
            if (detailBox && key && insightDetails[key]) {
                detailBox.textContent = insightDetails[key];
            }
            document.querySelectorAll("[data-sim-insight-chip]").forEach((item) => {
                item.classList.remove("bg-primary-container", "text-white", "border-primary-container");
                item.classList.add("bg-white", "text-slate-600", "border-slate-300");
            });
            chip.classList.remove("bg-white", "text-slate-600", "border-slate-300");
            chip.classList.add("bg-primary-container", "text-white", "border-primary-container");
        };
    });
}

function renderHypothesisPanel(payload, signalPayload) {
    const content = document.getElementById("sim-hypothesis-content");
    const meta = document.getElementById("sim-hypothesis-meta");
    if (!content) return;

    const items = (payload?.items || []).slice().sort((a, b) => a.horizon_years - b.horizon_years);
    if (!items.length) {
        content.innerHTML = `<div class="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600">Chưa có giả thuyết để hiển thị.</div>`;
        return;
    }

    const selected = items.find((item) => Number(item.horizon_years) === Number(activeHypothesisTab)) || items[0];
    activeHypothesisTab = Number(selected.horizon_years);
    const signalCount = signalPayload?.items?.length || 0;
    if (meta) {
        meta.textContent = `Nguồn hybrid external • ${signalCount} tín hiệu gần nhất • Độ tin cậy ${(Number(selected.confidence || 0) * 100).toFixed(1)}%`;
    }

    const drivers = (selected.drivers || []).map((driver) => {
        const effect = Number(driver.effect || 0);
        return `<li class="text-xs text-slate-600">${driver.factor}: <span class="font-bold ${effect >= 0 ? "text-emerald-600" : "text-rose-600"}">${effect >= 0 ? "+" : ""}${effect.toFixed(2)}pp</span></li>`;
    }).join("");
    const longform = Array.isArray(selected.longform_analysis) ? selected.longform_analysis : [];
    const tocButtons = longform.map((section, idx) => `
        <button type="button" data-hypothesis-section="${idx}" class="rounded-full border border-slate-300 bg-white px-3 py-1 text-[11px] font-semibold text-slate-600 hover:border-primary-container hover:text-primary-container transition">
            ${section.title || section.id || `Mục ${idx + 1}`}
        </button>
    `).join("");
    const longformSections = longform.map((section, idx) => `
        <article id="hypothesis-section-${idx}" data-hypothesis-article="${idx}" class="rounded-lg border border-slate-200 bg-white p-3 transition">
            <div class="flex items-center justify-between gap-3">
                <h4 class="text-xs font-black text-primary-container tracking-tight">${section.title || section.id || `Mục ${idx + 1}`}</h4>
                <button type="button" data-hypothesis-collapse="${idx}" class="rounded border border-slate-300 bg-white px-2 py-1 text-[10px] font-bold text-slate-600 hover:border-primary-container hover:text-primary-container transition">
                    Thu gọn
                </button>
            </div>
            <div data-hypothesis-body="${idx}" class="mt-2">
                <p class="text-xs text-slate-700 leading-relaxed whitespace-pre-line">${section.content || ""}</p>
                <div class="mt-2 flex flex-wrap items-center gap-2 text-[10px] text-slate-500">
                    <span class="rounded-full bg-slate-100 px-2 py-0.5">Confidence ${(Number(selected.confidence || 0) * 100).toFixed(1)}%</span>
                    <span class="rounded-full bg-slate-100 px-2 py-0.5">Source: hybrid simulation + external signals</span>
                </div>
            </div>
        </article>
    `).join("");

    content.innerHTML = `
        <div class="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <p class="text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-2">Giả thuyết chính</p>
            <p class="text-sm font-semibold text-slate-700 leading-relaxed">${selected.summary}</p>
            <div class="mt-3">
                <p class="text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-1">Drivers chính</p>
                <ul class="space-y-1">${drivers || "<li class='text-xs text-slate-500'>Chưa có driver.</li>"}</ul>
            </div>
        </div>
        <div class="space-y-3">
            <div class="rounded-lg border border-rose-100 bg-rose-50 p-3">
                <p class="text-[10px] uppercase tracking-wider text-rose-600 font-bold mb-1">Kịch bản xấu</p>
                <p class="text-xs text-rose-700 leading-relaxed">${selected.downside}</p>
            </div>
            <div class="rounded-lg border border-emerald-100 bg-emerald-50 p-3">
                <p class="text-[10px] uppercase tracking-wider text-emerald-600 font-bold mb-1">Kịch bản tốt</p>
                <p class="text-xs text-emerald-700 leading-relaxed">${selected.upside}</p>
            </div>
            <div class="rounded-lg border border-blue-100 bg-blue-50 p-3">
                <p class="text-[10px] uppercase tracking-wider text-blue-600 font-bold mb-1">Khuyến nghị điều hành</p>
                <p class="text-xs text-blue-700 leading-relaxed">${selected.recommendations}</p>
            </div>
            ${longform.length ? `
                <div class="sticky top-2 z-10 rounded-lg border border-slate-200 bg-slate-50/95 p-3 backdrop-blur">
                    <p class="text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-2">Mục lục phân tích chi tiết</p>
                    <div id="hypothesis-toc" class="flex flex-wrap gap-2">${tocButtons}</div>
                </div>
                <div id="hypothesis-longform" class="space-y-3">${longformSections}</div>
            ` : ""}
        </div>
    `;

    if (longform.length) {
        const tocNodes = Array.from(content.querySelectorAll("[data-hypothesis-section]"));
        const collapseState = {};
        content.querySelectorAll("[data-hypothesis-section]").forEach((btn) => {
            btn.addEventListener("click", () => {
                const idx = Number(btn.getAttribute("data-hypothesis-section") || 0);
                const target = content.querySelector(`#hypothesis-section-${idx}`);
                const body = content.querySelector(`[data-hypothesis-body="${idx}"]`);
                const collapseBtn = content.querySelector(`[data-hypothesis-collapse="${idx}"]`);
                if (body && collapseState[idx]) {
                    body.classList.remove("hidden");
                    collapseState[idx] = false;
                    if (collapseBtn) collapseBtn.textContent = "Thu gọn";
                }
                if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
            });
        });
        content.querySelectorAll("[data-hypothesis-collapse]").forEach((btn) => {
            btn.addEventListener("click", () => {
                const idx = Number(btn.getAttribute("data-hypothesis-collapse") || 0);
                const body = content.querySelector(`[data-hypothesis-body="${idx}"]`);
                const collapsed = Boolean(collapseState[idx]);
                if (!body) return;
                if (collapsed) {
                    body.classList.remove("hidden");
                    btn.textContent = "Thu gọn";
                } else {
                    body.classList.add("hidden");
                    btn.textContent = "Mở rộng";
                }
                collapseState[idx] = !collapsed;
            });
        });
        const applyActiveSection = (idx) => {
            tocNodes.forEach((node) => {
                const nodeIdx = Number(node.getAttribute("data-hypothesis-section") || -1);
                const isActive = nodeIdx === idx;
                node.classList.toggle("bg-primary-container", isActive);
                node.classList.toggle("text-white", isActive);
                node.classList.toggle("border-primary-container", isActive);
                node.classList.toggle("bg-white", !isActive);
                node.classList.toggle("text-slate-600", !isActive);
                node.classList.toggle("border-slate-300", !isActive);
            });
            content.querySelectorAll("[data-hypothesis-article]").forEach((article) => {
                const articleIdx = Number(article.getAttribute("data-hypothesis-article") || -1);
                const isActive = articleIdx === idx;
                article.classList.toggle("border-primary-container", isActive);
                article.classList.toggle("shadow-sm", isActive);
            });
        };
        const observer = new IntersectionObserver((entries) => {
            const visible = entries
                .filter((entry) => entry.isIntersecting)
                .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
            if (!visible.length) return;
            const activeId = visible[0].target.getAttribute("data-hypothesis-article");
            applyActiveSection(Number(activeId || 0));
        }, {
            root: null,
            rootMargin: "-20% 0px -60% 0px",
            threshold: [0.25, 0.5, 0.75],
        });
        content.querySelectorAll("[data-hypothesis-article]").forEach((article) => observer.observe(article));
        applyActiveSection(0);
    }

    document.querySelectorAll("[data-hypothesis-tab]").forEach((btn) => {
        const value = Number(btn.getAttribute("data-hypothesis-tab"));
        const isActive = value === activeHypothesisTab;
        btn.classList.toggle("bg-white", isActive);
        btn.classList.toggle("text-primary-container", isActive);
        btn.classList.toggle("border", isActive);
        btn.classList.toggle("border-slate-200", isActive);
        btn.classList.toggle("text-slate-600", !isActive);
    });
}

// ────────────────────────────────────────────────────────────
//  CHART 2: Risk Distribution Donut (FIX: sizing overflow)
// ────────────────────────────────────────────────────────────
function renderRiskDist(scenario) {
    const container = document.getElementById("sim-risk-dist");
    const canvas = document.getElementById("sim-risk-chart");
    if (!container || !canvas || !scenario?.risk_distribution) return;

    const dist = scenario.risk_distribution;
    const total = Object.values(dist).reduce((a, b) => a + b, 0);

    // BUG FIX: Use container dimensions properly
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const canvasH = Math.min(220, rect.height || 220);
    const cx = rect.width / 2;
    const cy = canvasH / 2;
    const radius = Math.min(cx, cy) * 0.65;

    canvas.width = rect.width * dpr;
    canvas.height = canvasH * dpr;
    canvas.style.width = rect.width + "px";
    canvas.style.height = canvasH + "px";
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, canvasH);

    // Draw Donut
    let startAngle = -Math.PI / 2;
    for (const [key, count] of Object.entries(dist)) {
        if (count === 0) continue;
        const sliceAngle = (count / total) * 2 * Math.PI;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, startAngle, startAngle + sliceAngle);
        ctx.lineWidth = radius * 0.4;
        ctx.strokeStyle = RISK_COLORS[key] || SIM_COLORS.slate;
        ctx.stroke();
        startAngle += sliceAngle;
    }

    // Draw center text
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "bold 22px Inter, sans-serif";
    ctx.fillStyle = SIM_COLORS.primary;
    ctx.fillText(total.toLocaleString(), cx, cy - 6);
    ctx.font = "10px Inter, sans-serif";
    ctx.fillStyle = SIM_COLORS.slate;
    ctx.fillText("Tổng DN", cx, cy + 12);

    // Render legend
    container.innerHTML = Object.entries(dist).map(([key, count]) => {
        const pct = (count / Math.max(1, total) * 100).toFixed(1);
        return `
            <div class="flex items-center gap-3 bg-slate-50 rounded-lg p-2 transition-transform duration-300 hover:scale-105 fade-in">
                <div class="w-2 h-8 rounded-full" style="background:${RISK_COLORS[key]}"></div>
                <div class="flex flex-col">
                    <span class="text-[10px] font-bold text-slate-500 uppercase">${RISK_LABELS[key] || key}</span>
                    <span class="text-sm font-black text-slate-800">${count.toLocaleString()} <span class="text-xs font-normal text-slate-400">(${pct}%)</span></span>
                </div>
            </div>`;
    }).join("");
}

// ────────────────────────────────────────────────────────────
//  CHART 3: Waterfall – Parameter Contributions
// ────────────────────────────────────────────────────────────
function renderWaterfallChart(data) {
    const canvas = document.getElementById("sim-waterfall-chart");
    if (!canvas || !data?.contributions) return;

    const items = (data.contributions || []).filter(c => c.contribution_pp !== 0);
    if (!items.length) {
        drawCanvasEmptyState(canvas, "Không có đóng góp đáng kể ở cấu hình hiện tại.");
        return;
    }

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const W = rect.width;
    const H = 260;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.scale(dpr, dpr);

    const pad = { top: 25, right: 20, bottom: 50, left: 100 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.fillStyle = SIM_COLORS.slateBg;
    ctx.fillRect(0, 0, W, H);

    // Calculate scale
    const maxAbs = Math.max(...items.map(c => Math.abs(c.contribution_pp)), 0.5);
    const xMin = -maxAbs * 1.3;
    const xMax = maxAbs * 1.3;
    const xScale = (v) => pad.left + ((v - xMin) / (xMax - xMin)) * plotW;
    const barH = Math.min(28, plotH / items.length - 4);

    // Zero line
    const zeroX = xScale(0);
    ctx.strokeStyle = SIM_COLORS.slateLight;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(zeroX, pad.top);
    ctx.lineTo(zeroX, pad.top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Bars
    items.forEach((c, i) => {
        const y = pad.top + (i / items.length) * plotH + (plotH / items.length - barH) / 2;
        const w = Math.abs(c.contribution_pp) / (xMax - xMin) * plotW;
        const x = c.contribution_pp >= 0 ? zeroX : zeroX - w;
        const color = c.contribution_pp > 0 ? SIM_COLORS.rose : SIM_COLORS.emerald;

        // Bar
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(x, y, w, barH, 4);
        ctx.fill();

        // Value label
        ctx.fillStyle = color;
        ctx.font = "bold 10px Inter, sans-serif";
        ctx.textAlign = c.contribution_pp >= 0 ? "left" : "right";
        const labelX = c.contribution_pp >= 0 ? x + w + 5 : x - 5;
        ctx.fillText(`${c.contribution_pp > 0 ? "+" : ""}${c.contribution_pp.toFixed(2)}pp`, labelX, y + barH / 2 + 3);

        // Label
        ctx.fillStyle = SIM_COLORS.primary;
        ctx.font = "11px Inter, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(c.label, pad.left - 8, y + barH / 2 + 3);
    });

    // Title delineation
    ctx.fillStyle = SIM_COLORS.slateLight;
    ctx.font = "9px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("← Giảm nợ đọng", zeroX - plotW * 0.25, H - 8);
    ctx.fillText("Tăng nợ đọng →", zeroX + plotW * 0.25, H - 8);
}

// ────────────────────────────────────────────────────────────
//  CHART 4: Tornado – Sensitivity Analysis
// ────────────────────────────────────────────────────────────
function renderTornadoChart(data) {
    const canvas = document.getElementById("sim-tornado-chart");
    if (!canvas || !data?.items) return;

    const items = (data.items || []).slice().sort((a, b) => b.sensitivity_range - a.sensitivity_range);
    if (!items.length) {
        drawCanvasEmptyState(canvas, "Không có dữ liệu độ nhạy cho cấu hình hiện tại.", 280);
        return;
    }
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const W = rect.width;
    const H = 280;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.scale(dpr, dpr);

    const pad = { top: 25, right: 25, bottom: 35, left: 110 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;
    const baseRate = data.baseline_delinquency_rate;

    ctx.fillStyle = SIM_COLORS.slateBg;
    ctx.fillRect(0, 0, W, H);

    // Scale: symmetric around baseline
    const maxDelta = Math.max(...items.map(i => Math.max(Math.abs(i.min_delinq_rate - baseRate), Math.abs(i.max_delinq_rate - baseRate))), 2);
    const xMin = baseRate - maxDelta * 1.4;
    const xMax = baseRate + maxDelta * 1.4;
    const xScale = (v) => pad.left + ((v - xMin) / (xMax - xMin)) * plotW;

    const barH = Math.min(30, plotH / items.length - 6);

    // Center line (baseline)
    const centerX = xScale(baseRate);
    ctx.strokeStyle = SIM_COLORS.primary;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    ctx.moveTo(centerX, pad.top - 5);
    ctx.lineTo(centerX, pad.top + plotH + 5);
    ctx.stroke();
    ctx.setLineDash([]);

    // Baseline label
    ctx.fillStyle = SIM_COLORS.primary;
    ctx.font = "bold 9px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`Baseline ${baseRate}%`, centerX, pad.top - 8);

    items.forEach((item, i) => {
        const y = pad.top + (i / items.length) * plotH + (plotH / items.length - barH) / 2;

        // Low bar (min range)
        const x1 = xScale(Math.min(item.min_delinq_rate, item.max_delinq_rate));
        const x2 = xScale(Math.max(item.min_delinq_rate, item.max_delinq_rate));

        // Left half
        const leftX = Math.min(x1, centerX);
        const leftW = centerX - leftX;
        if (leftW > 0) {
            ctx.fillStyle = SIM_COLORS.emerald + "CC";
            ctx.beginPath();
            ctx.roundRect(leftX, y, leftW, barH, [4, 0, 0, 4]);
            ctx.fill();
        }

        // Right half
        const rightX = centerX;
        const rightW = Math.max(x2, centerX) - centerX;
        if (rightW > 0) {
            ctx.fillStyle = SIM_COLORS.rose + "CC";
            ctx.beginPath();
            ctx.roundRect(rightX, y, rightW, barH, [0, 4, 4, 0]);
            ctx.fill();
        }

        // Range label
        ctx.fillStyle = SIM_COLORS.slate;
        ctx.font = "bold 9px Inter, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(`${item.sensitivity_range.toFixed(2)}pp`, x2 + 6, y + barH / 2 + 3);

        // Parameter label
        ctx.fillStyle = SIM_COLORS.primary;
        ctx.font = "11px Inter, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(item.label, pad.left - 8, y + barH / 2 + 3);
    });
}

// ────────────────────────────────────────────────────────────
//  CHART 5: Radar – Scenario vs Baseline
// ────────────────────────────────────────────────────────────
function renderRadarChart(scenario, params) {
    const canvas = document.getElementById("sim-radar-chart");
    if (!canvas || !params) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const size = Math.min(rect.width, 280);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = size + "px";
    canvas.style.height = size + "px";
    ctx.scale(dpr, dpr);

    const cx = size / 2;
    const cy = size / 2;
    const R = size * 0.35;

    const axes = [
        { label: "VAT", value: params.vat_rate / 25, baseline: 10 / 25 },
        { label: "CIT", value: params.cit_rate / 40, baseline: 20 / 40 },
        { label: "Thanh tra", value: params.audit_coverage_pct / 50, baseline: 5 / 50 },
        { label: "Phạt", value: params.penalty_multiplier / 5, baseline: 1 / 5 },
        { label: "Lãi suất", value: params.interest_rate / 25, baseline: 6 / 25 },
        { label: "GDP", value: (params.economic_growth_pct + 5) / 20, baseline: (6.5 + 5) / 20 },
        { label: "CPI", value: params.cpi_pct / 20, baseline: 3.5 / 20 },
        { label: "Thất nghiệp", value: params.unemployment_pct / 25, baseline: 2.3 / 25 },
        { label: "Tỷ giá", value: (params.exchange_rate_delta_pct + 15) / 30, baseline: 0.5 },
        { label: "Kỳ hạn", value: params.projection_years / 5, baseline: 2 / 5 },
    ];
    const n = axes.length;
    const angleStep = (2 * Math.PI) / n;

    ctx.fillStyle = SIM_COLORS.slateBg;
    ctx.fillRect(0, 0, size, size);

    // Grid circles
    for (let r = 0.25; r <= 1; r += 0.25) {
        ctx.beginPath();
        for (let i = 0; i <= n; i++) {
            const angle = -Math.PI / 2 + i * angleStep;
            const x = cx + Math.cos(angle) * R * r;
            const y = cy + Math.sin(angle) * R * r;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.strokeStyle = SIM_COLORS.gridLine;
        ctx.lineWidth = 0.5;
        ctx.stroke();
    }

    // Axis lines + labels
    axes.forEach((ax, i) => {
        const angle = -Math.PI / 2 + i * angleStep;
        const x = cx + Math.cos(angle) * R;
        const y = cy + Math.sin(angle) * R;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x, y);
        ctx.strokeStyle = SIM_COLORS.gridLine;
        ctx.lineWidth = 0.5;
        ctx.stroke();

        const lx = cx + Math.cos(angle) * (R + 18);
        const ly = cy + Math.sin(angle) * (R + 18);
        ctx.fillStyle = SIM_COLORS.primary;
        ctx.font = "bold 9px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(ax.label, lx, ly);
    });

    // Draw polygon helper
    function drawPolygon(values, fillColor, strokeColor) {
        ctx.beginPath();
        values.forEach((v, i) => {
            const angle = -Math.PI / 2 + i * angleStep;
            const r = Math.max(0, Math.min(1, v)) * R;
            const x = cx + Math.cos(angle) * r;
            const y = cy + Math.sin(angle) * r;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Baseline
    drawPolygon(axes.map(a => a.baseline), "rgba(148, 163, 184, 0.15)", SIM_COLORS.slateLight);
    // Current
    drawPolygon(axes.map(a => a.value), "rgba(2, 132, 199, 0.2)", SIM_COLORS.sky);

    // Dots for current
    axes.forEach((ax, i) => {
        const angle = -Math.PI / 2 + i * angleStep;
        const r = Math.max(0, Math.min(1, ax.value)) * R;
        ctx.beginPath();
        ctx.arc(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r, 3, 0, Math.PI * 2);
        ctx.fillStyle = SIM_COLORS.sky;
        ctx.fill();
    });
}

// ────────────────────────────────────────────────────────────
//  CHART 6: Bubble – Industry Impact Map
// ────────────────────────────────────────────────────────────
function renderBubbleChart(scenario) {
    const canvas = document.getElementById("sim-bubble-chart");
    if (!canvas || !scenario?.industry_impacts?.length) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const W = rect.width;
    const H = 280;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.scale(dpr, dpr);

    const pad = { top: 25, right: 30, bottom: 45, left: 65 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.fillStyle = SIM_COLORS.slateBg;
    ctx.fillRect(0, 0, W, H);

    const data = scenario.industry_impacts;
    const xVals = data.map(d => d.baseline_delinquency_rate);
    const yVals = data.map(d => d.simulated_delinquency_rate);
    const counts = data.map(d => d.company_count);

    const xMin = Math.max(0, Math.min(...xVals) - 2);
    const xMax = Math.max(...xVals) + 2;
    const yMin = Math.max(0, Math.min(...yVals) - 2);
    const yMax = Math.max(...yVals) + 2;
    const maxCount = Math.max(...counts);

    const xs = (v) => pad.left + ((v - xMin) / (xMax - xMin)) * plotW;
    const ys = (v) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    // Grid
    ctx.strokeStyle = SIM_COLORS.gridLine;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (i / 4) * plotH;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
        const val = yMax - (i / 4) * (yMax - yMin);
        ctx.fillStyle = SIM_COLORS.slateLight; ctx.font = "9px Inter"; ctx.textAlign = "right";
        ctx.fillText(val.toFixed(1) + "%", pad.left - 6, y + 3);
    }

    // Diagonal (no-change line)
    ctx.strokeStyle = SIM_COLORS.slateLight;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    const diagMin = Math.max(xMin, yMin);
    const diagMax = Math.min(xMax, yMax);
    ctx.moveTo(xs(diagMin), ys(diagMin));
    ctx.lineTo(xs(diagMax), ys(diagMax));
    ctx.stroke();
    ctx.setLineDash([]);

    // Axis labels
    ctx.fillStyle = SIM_COLORS.slateLight;
    ctx.font = "9px Inter";
    ctx.textAlign = "center";
    ctx.fillText("Baseline Rate (%)", pad.left + plotW / 2, H - 5);
    ctx.save();
    ctx.translate(12, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Simulated Rate (%)", 0, 0);
    ctx.restore();

    // Colors for industries
    const bubbleColors = ["#0284c7", "#7c3aed", "#f59e0b", "#10b981", "#ef4444", "#f97316", "#06b6d4", "#8b5cf6", "#ec4899", "#14b8a6", "#84cc16", "#6366f1"];

    // Bubbles
    data.forEach((d, i) => {
        const bx = xs(d.baseline_delinquency_rate);
        const by = ys(d.simulated_delinquency_rate);
        const r = Math.max(6, Math.min(28, (d.company_count / maxCount) * 28));
        const color = bubbleColors[i % bubbleColors.length];

        ctx.beginPath();
        ctx.arc(bx, by, r, 0, Math.PI * 2);
        ctx.fillStyle = color + "55";
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label
        if (r > 8) {
            const shortName = d.industry.length > 8 ? d.industry.substring(0, 7) + "…" : d.industry;
            ctx.fillStyle = SIM_COLORS.primary;
            ctx.font = "bold 8px Inter";
            ctx.textAlign = "center";
            ctx.fillText(shortName, bx, by + r + 12);
        }
    });
}

// ────────────────────────────────────────────────────────────
//  CHART 7: Gauge – Overall Risk Meter
// ────────────────────────────────────────────────────────────
function renderGauge(scenario) {
    const canvas = document.getElementById("sim-gauge-chart");
    if (!canvas || !scenario) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const size = Math.min(rect.width, 200);
    canvas.width = size * dpr;
    canvas.height = (size * 0.65) * dpr;
    canvas.style.width = size + "px";
    canvas.style.height = (size * 0.65) + "px";
    ctx.scale(dpr, dpr);

    const W = size;
    const H = size * 0.65;
    const cx = W / 2;
    const cy = H - 15;
    const R = Math.min(cx - 10, cy - 10);
    if (!Number.isFinite(R) || R <= 5) {
        drawCanvasEmptyState(canvas, "Không đủ không gian để hiển thị Gauge.", Math.max(140, H));
        return;
    }

    ctx.fillStyle = SIM_COLORS.slateBg;
    ctx.fillRect(0, 0, W, H);

    // Gauge arc background segments
    const segments = [
        { start: Math.PI, end: Math.PI + Math.PI * 0.25, color: SIM_COLORS.emerald },
        { start: Math.PI + Math.PI * 0.25, end: Math.PI + Math.PI * 0.5, color: SIM_COLORS.amber },
        { start: Math.PI + Math.PI * 0.5, end: Math.PI + Math.PI * 0.75, color: SIM_COLORS.orange },
        { start: Math.PI + Math.PI * 0.75, end: Math.PI * 2, color: SIM_COLORS.rose },
    ];

    segments.forEach(seg => {
        ctx.beginPath();
        ctx.arc(cx, cy, R, seg.start, seg.end);
        ctx.lineWidth = 16;
        ctx.strokeStyle = seg.color + "44";
        ctx.lineCap = "butt";
        ctx.stroke();
    });

    // Active arc
    const rate = scenario.simulated_delinquency_rate / 100;
    const angle = Math.PI + rate * Math.PI;
    const activeColor = rate > 0.6 ? SIM_COLORS.rose : rate > 0.4 ? SIM_COLORS.orange : rate > 0.2 ? SIM_COLORS.amber : SIM_COLORS.emerald;
    ctx.beginPath();
    ctx.arc(cx, cy, R, Math.PI, Math.min(angle, Math.PI * 2));
    ctx.lineWidth = 16;
    ctx.strokeStyle = activeColor;
    ctx.lineCap = "round";
    ctx.stroke();

    // Needle
    const needleAngle = Math.PI + rate * Math.PI;
    const needleLen = R - 22;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(needleAngle) * needleLen, cy + Math.sin(needleAngle) * needleLen);
    ctx.strokeStyle = SIM_COLORS.primary;
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.stroke();

    // Center dot
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = SIM_COLORS.primary;
    ctx.fill();

    // Value text
    ctx.textAlign = "center";
    ctx.font = "bold 18px Inter";
    ctx.fillStyle = activeColor;
    ctx.fillText(`${scenario.simulated_delinquency_rate}%`, cx, cy - 18);
    ctx.font = "9px Inter";
    ctx.fillStyle = SIM_COLORS.slateLight;
    ctx.fillText("Tỷ lệ nợ đọng", cx, cy - 5);
}

// ────────────────────────────────────────────────────────────
//  CHART 8: Heatmap – Industry × Risk Level
// ────────────────────────────────────────────────────────────
function renderHeatmap(data) {
    const canvas = document.getElementById("sim-heatmap-chart");
    if (!canvas || !data?.cells?.length) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const W = rect.width;

    const industries = data.industries || [];
    const levels = data.risk_levels || ["low", "medium", "high", "critical"];
    const rows = industries.length;
    const cols = levels.length;
    if (!rows || !cols) return;

    const cellH = 32;
    const pad = { top: 30, right: 15, bottom: 15, left: 120 };
    const H = pad.top + rows * cellH + pad.bottom;
    const cellW = (W - pad.left - pad.right) / cols;

    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.scale(dpr, dpr);
    ctx.fillStyle = SIM_COLORS.slateBg;
    ctx.fillRect(0, 0, W, H);

    // Column headers
    const levelLabels = { low: "Thấp", medium: "TB", high: "Cao", critical: "Nguy hiểm" };
    levels.forEach((lev, j) => {
        const x = pad.left + j * cellW + cellW / 2;
        ctx.fillStyle = SIM_COLORS.primary;
        ctx.font = "bold 9px Inter";
        ctx.textAlign = "center";
        ctx.fillText(levelLabels[lev] || lev, x, pad.top - 10);
    });

    // Build lookup
    const cellMap = {};
    data.cells.forEach(c => { cellMap[`${c.industry}_${c.risk_level}`] = c; });

    // Draw cells
    industries.forEach((ind, i) => {
        const y = pad.top + i * cellH;

        // Row label
        const shortInd = ind.length > 15 ? ind.substring(0, 14) + "…" : ind;
        ctx.fillStyle = SIM_COLORS.primary;
        ctx.font = "10px Inter";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(shortInd, pad.left - 8, y + cellH / 2);

        levels.forEach((lev, j) => {
            const x = pad.left + j * cellW;
            const cell = cellMap[`${ind}_${lev}`];
            const pct = cell?.percentage || 0;

            // Color intensity
            const maxColor = RISK_COLORS[lev] || SIM_COLORS.slate;
            const alpha = Math.max(0.08, Math.min(0.85, pct / 100));
            ctx.fillStyle = maxColor;
            ctx.globalAlpha = alpha;
            ctx.beginPath();
            ctx.roundRect(x + 2, y + 2, cellW - 4, cellH - 4, 4);
            ctx.fill();
            ctx.globalAlpha = 1;

            // Count text
            if (cell?.count > 0) {
                ctx.fillStyle = alpha > 0.5 ? SIM_COLORS.white : SIM_COLORS.primary;
                ctx.font = "bold 10px Inter";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(cell.count.toString(), x + cellW / 2, y + cellH / 2);
            }
        });
    });
}

// ────────────────────────────────────────────────────────────
//  CHART 9: Slope – Industry Rank Change
// ────────────────────────────────────────────────────────────
function renderSlopeChart(scenario) {
    const canvas = document.getElementById("sim-slope-chart");
    if (!canvas || !scenario?.industry_impacts?.length) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const W = rect.width;
    const H = 280;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.scale(dpr, dpr);

    ctx.fillStyle = SIM_COLORS.slateBg;
    ctx.fillRect(0, 0, W, H);

    const pad = { top: 30, right: 100, bottom: 20, left: 100 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const data = [...scenario.industry_impacts].slice(0, 10);
    const baselineRanked = [...data].sort((a, b) => b.baseline_delinquency_rate - a.baseline_delinquency_rate);
    const simulatedRanked = [...data].sort((a, b) => b.simulated_delinquency_rate - a.simulated_delinquency_rate);

    const baseRankMap = {};
    baselineRanked.forEach((d, i) => { baseRankMap[d.industry] = i; });
    const simRankMap = {};
    simulatedRanked.forEach((d, i) => { simRankMap[d.industry] = i; });

    const n = data.length;
    const yStep = plotH / Math.max(1, n - 1);

    const slopeColors = ["#0284c7", "#7c3aed", "#f59e0b", "#10b981", "#ef4444", "#f97316", "#06b6d4", "#8b5cf6", "#ec4899", "#14b8a6"];

    // Column headers
    ctx.fillStyle = SIM_COLORS.primary;
    ctx.font = "bold 10px Inter";
    ctx.textAlign = "center";
    ctx.fillText("Baseline", pad.left - 30, pad.top - 12);
    ctx.fillText("Mô phỏng", W - pad.right + 30, pad.top - 12);

    data.forEach((d, idx) => {
        const baseRank = baseRankMap[d.industry];
        const simRank = simRankMap[d.industry];
        const color = slopeColors[idx % slopeColors.length];

        const y1 = pad.top + baseRank * yStep;
        const y2 = pad.top + simRank * yStep;

        // Line
        ctx.beginPath();
        ctx.moveTo(pad.left, y1);
        ctx.lineTo(W - pad.right, y2);
        ctx.strokeStyle = color + "AA";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Dots
        [{ x: pad.left, y: y1 }, { x: W - pad.right, y: y2 }].forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        });

        // Labels
        const shortName = d.industry.length > 10 ? d.industry.substring(0, 9) + "…" : d.industry;
        ctx.fillStyle = SIM_COLORS.primary;
        ctx.font = "9px Inter";
        ctx.textAlign = "right";
        ctx.fillText(`${shortName} ${d.baseline_delinquency_rate}%`, pad.left - 8, y1 + 3);
        ctx.textAlign = "left";
        ctx.fillText(`${d.simulated_delinquency_rate}% ${shortName}`, W - pad.right + 8, y2 + 3);
    });
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

// ── Events ─────────────────────────────────────────────────
function bindEvents() {
    const runBtn = document.getElementById("sim-run-btn");
    if (runBtn) runBtn.addEventListener("click", () => runSimulation("Custom Scenario"));

    const resetBtn = document.getElementById("sim-reset-btn");
    if (resetBtn) resetBtn.addEventListener("click", () => {
        applyPresetToSliders({
            vat_rate: 10, cit_rate: 20, audit_coverage_pct: 5,
            penalty_multiplier: 1, interest_rate: 6, economic_growth_pct: 6.5,
            cpi_pct: 3.5, unemployment_pct: 2.3, exchange_rate_delta_pct: 0, projection_years: 5
        });
        if (baselineData) {
            renderKPIs(baselineData, null);
            runSimulation("Baseline");
        }
    });

    window.addEventListener("resize", debounce(() => {
        if (currentScenario) {
            renderChart(currentScenario);
            renderRiskDist(currentScenario);
            renderGauge(currentScenario);
            renderBubbleChart(currentScenario);
            renderSlopeChart(currentScenario);
            renderRadarChart(currentScenario, gatherParams());
            renderProjectionInsights(currentScenario);
            renderHealthGauge(currentScenario.scenario_health_score);
            renderStackedAreaChart(currentScenario);
            if (contributionsData) renderWaterfallChart(contributionsData);
            if (sensitivityData) renderTornadoChart(sensitivityData);
            if (heatmapData) renderHeatmap(heatmapData);
            // Historical is static sizing, handled by its render function
        }
    }, 250));

    const projYears = document.getElementById("sim-projection-years");
    if (projYears) {
        projYears.addEventListener("change", debouncedRunSimulation);
    }

    document.querySelectorAll('input[name="sim-historical-mode"]').forEach((modeInput) => {
        modeInput.addEventListener("change", () => {
            if (historicalData) renderHistoricalChart(historicalData, currentScenario);
        });
    });

    document.querySelectorAll("[data-hypothesis-tab]").forEach((btn) => {
        btn.addEventListener("click", () => {
            activeHypothesisTab = Number(btn.getAttribute("data-hypothesis-tab") || 1);
            if (hypothesisPayload) {
                renderHypothesisPanel(hypothesisPayload, externalSignalPayload);
            }
        });
    });
}

// ────────────────────────────────────────────────────────────
//  CHART 11: Health Score Gauge
// ────────────────────────────────────────────────────────────
function renderHealthGauge(score) {
    const valSpan = document.getElementById("sim-health-score-val");
    if (valSpan) valSpan.textContent = score.toFixed(1);

    const canvas = document.getElementById("sim-health-gauge");
    if (!canvas) return;
    if (!hasChartJs()) {
        drawCanvasEmptyState(canvas, "Thiếu Chart.js để render Gauge.", 280);
        return;
    }
    
    // Destroy previous Chart instance if exists
    if (charts["health"]) charts["health"].destroy();
    
    // Determine color based on score
    let color = SIM_COLORS.rose;
    if (score >= 40) color = SIM_COLORS.amber;
    if (score >= 70) color = SIM_COLORS.emerald;
    if (score >= 85) color = SIM_COLORS.sky;

    charts["health"] = new Chart(canvas, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [score, 100 - score],
                backgroundColor: [color, SIM_COLORS.slateBg],
                borderWidth: 0,
                cutout: '80%',
                circumference: 180,
                rotation: 270
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: { enabled: false },
                legend: { display: false }
            },
            animation: {
                animateScale: true,
                animateRotate: true,
                duration: 1500,
                easing: 'easeOutBounce'
            }
        }
    });
}

// ────────────────────────────────────────────────────────────
//  CHART 12: Monte Carlo Fan Chart
// ────────────────────────────────────────────────────────────
function renderMonteCarloChart(data) {
    const canvas = document.getElementById("sim-montecarlo-chart");
    if (!canvas || !data.bands) return;
    if (!hasChartJs()) {
        drawCanvasEmptyState(canvas, "Thiếu Chart.js để render Monte Carlo.", 280);
        return;
    }
    
    if (charts["montecarlo"]) charts["montecarlo"].destroy();

    const labels = data.bands.map(b => b.quarter);
    const p10 = data.bands.map(b => b.p10);
    const p25 = data.bands.map(b => b.p25);
    const p50 = data.bands.map(b => b.p50);
    const p75 = data.bands.map(b => b.p75);
    const p90 = data.bands.map(b => b.p90);
    const base = data.bands.map(b => b.baseline);

    charts["montecarlo"] = new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'P90 (Top 10%)',
                    data: p90,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(2, 132, 199, 0.1)',
                    fill: '+1',
                    pointRadius: 0
                },
                {
                    label: 'P75',
                    data: p75,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(2, 132, 199, 0.2)',
                    fill: '+1',
                    pointRadius: 0
                },
                {
                    label: 'P50 (Median)',
                    data: p50,
                    borderColor: SIM_COLORS.primary,
                    borderWidth: 2,
                    backgroundColor: 'transparent',
                    pointRadius: 2,
                    fill: false
                },
                {
                    label: 'P25',
                    data: p25,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(2, 132, 199, 0.2)',
                    fill: '-1',
                    pointRadius: 0
                },
                {
                    label: 'P10',
                    data: p10,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(2, 132, 199, 0.1)',
                    fill: '-1',
                    pointRadius: 0
                },
                {
                    label: 'Baseline',
                    data: base,
                    borderColor: SIM_COLORS.slateLight,
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: { display: true, title: { display: true, text: "Doanh thu (Tỷ VNĐ)" } },
                x: { grid: { display: false } }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(ctx) { return ctx.dataset.label + ": " + ctx.raw + " tỷ"; }
                    }
                }
            }
        }
    });
}

// ────────────────────────────────────────────────────────────
//  CHART 13: Historical Comparison
// ────────────────────────────────────────────────────────────
function getHistoricalMode() {
    const selected = document.querySelector('input[name="sim-historical-mode"]:checked');
    return selected?.value || "historical_only";
}

function renderHistoricalChart(histData, scenario = null) {
    const canvas = document.getElementById("sim-historical-chart");
    if (!canvas || !histData.revenue_trend) return;
    if (!hasChartJs()) {
        drawCanvasEmptyState(canvas, "Thiếu Chart.js để render Historical.", 280);
        return;
    }
    
    if (charts["historical"]) charts["historical"].destroy();

    const revData = histData.revenue_trend;
    const labels = revData.map(d => d.quarter);
    const revenue = revData.map(d => d.total_revenue);

    const mode = getHistoricalMode();
    const datasets = [{
        type: "bar",
        label: "Doanh thu Thực tế (Tổng Cục)",
        data: revenue,
        backgroundColor: "rgba(16, 185, 129, 0.2)",
        borderColor: SIM_COLORS.emerald,
        borderWidth: 1,
        borderRadius: 4
    }];
    if (mode === "historical_vs_scenario" && scenario?.quarterly_projection?.length) {
        datasets.push({
            type: "line",
            label: "Dự phóng Kịch bản",
            data: scenario.quarterly_projection.slice(0, labels.length).map((q) => q.simulated_value),
            borderColor: SIM_COLORS.primary,
            borderWidth: 2,
            pointRadius: 2,
            fill: false,
            tension: 0.25
        });
    }

    charts["historical"] = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, title: { display: true, text: "Tỷ VNĐ" } },
                x: { grid: { display: false } }
            },
            plugins: { legend: { position: 'bottom' } }
        }
    });
}

// ────────────────────────────────────────────────────────────
//  CHART 14: Stacked Area (Industry Revenue Breakdown)
// ────────────────────────────────────────────────────────────
function renderStackedAreaChart(scenario) {
    const canvas = document.getElementById("sim-stacked-area-chart");
    if (!canvas || !scenario.industry_impacts) return;
    if (!hasChartJs()) {
        drawCanvasEmptyState(canvas, "Thiếu Chart.js để render Stacked Area.", 280);
        return;
    }
    
    if (charts["stackedarea"]) charts["stackedarea"].destroy();

    const topIndustries = scenario.industry_impacts
        .sort((a, b) => b.company_count - a.company_count)
        .slice(0, 6)
        .map(i => i.industry);

    const labels = scenario.quarterly_projection.map(q => q.quarter);
    
    const colors = [SIM_COLORS.primary, SIM_COLORS.sky, SIM_COLORS.emerald, SIM_COLORS.amber, SIM_COLORS.rose, SIM_COLORS.violet];
    
    const datasets = topIndustries.map((ind, idx) => {
        const impact = scenario.industry_impacts.find(i => i.industry === ind);
        const ratio = impact.company_count / scenario.baseline_total_companies;
        return {
            label: ind,
            data: scenario.quarterly_projection.map(q => q.simulated_value * ratio),
            backgroundColor: `${colors[idx]}33`, // 20% opacity
            borderColor: colors[idx],
            borderWidth: 1,
            fill: true
        };
    });

    charts["stackedarea"] = new Chart(canvas, {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: { stacked: true, title: { display: true, text: "Tỷ VNĐ" } },
                x: { grid: { display: false } }
            },
            plugins: {
                legend: { position: 'right', labels: { boxWidth: 10, font: { size: 10 } } },
                filler: { propagate: false }
            }
        }
    });
}
