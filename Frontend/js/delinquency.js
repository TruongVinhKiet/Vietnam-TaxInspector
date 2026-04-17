/**
 * delinquency.js
 * ==============
 * Delinquency list page controller.
 * - Renders record cards from /api/delinquency
 * - Uses URL state for stable pagination/filter navigation
 * - Routes detail action to dedicated detail page
 */

const PAGE_SIZE = 25;
const ALLOWED_FRESHNESS = new Set(["fresh", "aging", "stale", "unknown"]);

const state = {
    currentPage: 1,
    totalPages: 1,
    total: 0,
    freshnessFilter: "",
};

let latestRequestId = 0;


function escapeHtml(value) {
    const str = value === null || value === undefined ? "" : String(value);
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;")
        .replace(/'/g, "&#39;");
}


function toSafeNumber(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}


function formatCurrencyCompact(value) {
    const num = toSafeNumber(value, 0);
    if (num >= 1e9) return `${(num / 1e9).toFixed(1)} Tỷ`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)} Tr`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(0)} K`;
    return num.toLocaleString("vi-VN");
}


function normalizeFreshnessValue(value) {
    const normalized = (value || "").toString().trim().toLowerCase();
    return ALLOWED_FRESHNESS.has(normalized) ? normalized : "";
}


function parseInitialStateFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const rawPage = Number(params.get("page") || "1");
    const parsedPage = Number.isInteger(rawPage) && rawPage > 0 ? rawPage : 1;

    return {
        page: parsedPage,
        freshness: normalizeFreshnessValue(params.get("freshness")),
    };
}


function syncUrlState() {
    const params = new URLSearchParams(window.location.search);

    if (state.currentPage > 1) {
        params.set("page", String(state.currentPage));
    } else {
        params.delete("page");
    }

    if (state.freshnessFilter) {
        params.set("freshness", state.freshnessFilter);
    } else {
        params.delete("freshness");
    }

    const query = params.toString();
    const nextUrl = `${window.location.pathname}${query ? `?${query}` : ""}`;
    window.history.replaceState({}, "", nextUrl);
}


function getRiskColor(probability) {
    if (probability > 0.8) return { bg: "bg-red-500", text: "bg-red-100 text-red-700", label: "Rất cao" };
    if (probability > 0.6) return { bg: "bg-orange-500", text: "bg-orange-100 text-orange-700", label: "Cao" };
    if (probability > 0.4) return { bg: "bg-yellow-500", text: "bg-yellow-700 bg-yellow-100", label: "Trung bình" };
    if (probability > 0.2) return { bg: "bg-blue-400", text: "bg-blue-100 text-blue-700", label: "Thấp" };
    return { bg: "bg-emerald-400", text: "bg-emerald-100 text-emerald-700", label: "Rất thấp" };
}


function getFreshnessBadge(freshness) {
    const key = (freshness || "unknown").toString().toLowerCase();
    if (key === "fresh") {
        return { label: "Fresh", classes: "bg-emerald-50 text-emerald-700 border border-emerald-100" };
    }
    if (key === "aging") {
        return { label: "Aging", classes: "bg-amber-50 text-amber-700 border border-amber-100" };
    }
    if (key === "stale") {
        return { label: "Stale", classes: "bg-rose-50 text-rose-700 border border-rose-100" };
    }
    return { label: "Unknown", classes: "bg-slate-100 text-slate-600 border border-slate-200" };
}


function getSourceBadge(scoreSource) {
    const key = (scoreSource || "unknown").toString().toLowerCase();
    if (key === "ml_model") {
        return { label: "ML", classes: "bg-blue-50 text-blue-700 border border-blue-100" };
    }
    if (key === "statistical_baseline") {
        return { label: "Baseline", classes: "bg-violet-50 text-violet-700 border border-violet-100" };
    }
    if (key === "no_data") {
        return { label: "No data", classes: "bg-slate-100 text-slate-600 border border-slate-200" };
    }
    return { label: "Fallback", classes: "bg-orange-50 text-orange-700 border border-orange-100" };
}


function buildDetailUrl(taxCode) {
    const params = new URLSearchParams();
    params.set("tax_code", taxCode);
    params.set("page", String(state.currentPage));
    if (state.freshnessFilter) {
        params.set("freshness", state.freshnessFilter);
    }
    return `delinquency-detail.html?${params.toString()}`;
}


function renderPredictionMeta(item) {
    const freshness = getFreshnessBadge(item.freshness);
    const source = getSourceBadge(item.score_source);
    const ageDays = Number.isFinite(Number(item.prediction_age_days)) ? Number(item.prediction_age_days) : null;
    const ageText = ageDays === null ? "N/A" : `${ageDays} ngày`;

    return `
        <div class="flex flex-wrap items-center gap-1.5 mt-2">
            <span class="text-[10px] px-2 py-0.5 rounded-full font-bold ${freshness.classes}">${freshness.label}</span>
            <span class="text-[10px] px-2 py-0.5 rounded-full font-bold ${source.classes}">${source.label}</span>
            <span class="text-[10px] text-slate-500 font-semibold">Age: ${escapeHtml(ageText)}</span>
            ${item.monotonic_adjusted ? '<span class="text-[10px] px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-semibold">mono-fix</span>' : ""}
        </div>`;
}


function renderProbabilityBreakdown(item) {
    const p30 = toSafeNumber(item.prob_30d, 0);
    const p60 = toSafeNumber(item.prob_60d, 0);
    const p90 = toSafeNumber(item.prob_90d, 0);

    return `
        <div class="flex gap-1.5 mt-2 flex-wrap">
            <span class="text-[10px] px-2 py-0.5 rounded ${p30 > 0.5 ? "bg-red-50 text-red-600" : "bg-slate-50 text-slate-500"} font-bold">
                30d: ${(p30 * 100).toFixed(0)}%
            </span>
            <span class="text-[10px] px-2 py-0.5 rounded ${p60 > 0.5 ? "bg-orange-50 text-orange-600" : "bg-slate-50 text-slate-500"} font-bold">
                60d: ${(p60 * 100).toFixed(0)}%
            </span>
            <span class="text-[10px] px-2 py-0.5 rounded ${p90 > 0.5 ? "bg-yellow-50 text-yellow-700" : "bg-slate-50 text-slate-500"} font-bold">
                90d: ${(p90 * 100).toFixed(0)}%
            </span>
        </div>`;
}


function renderTopReasons(reasons) {
    if (!Array.isArray(reasons) || reasons.length === 0) return "";

    const items = reasons.slice(0, 2).map((reasonObj) => {
        const reason = escapeHtml(reasonObj.reason || "");
        const weight = Math.max(0, Math.min(1, toSafeNumber(reasonObj.weight, 0)));
        const barWidth = Math.max(4, weight * 100);

        return `
            <div class="flex items-center gap-2 mt-1">
                <div class="w-12 h-1.5 bg-slate-100 rounded-full overflow-hidden flex-shrink-0">
                    <div class="h-full bg-primary-container/60 rounded-full" style="width:${barWidth}%"></div>
                </div>
                <span class="text-[10px] text-slate-500 leading-tight">${reason}</span>
            </div>`;
    }).join("");

    return `<div class="mt-1">${items}</div>`;
}


function renderPaymentSummary(summary) {
    if (!summary || !summary.data_available) return "";

    const onTime = toSafeNumber(summary.on_time_count, 0);
    const late = toSafeNumber(summary.late_count, 0);
    const unpaid = toSafeNumber(summary.unpaid_count, 0);

    return `
        <div class="flex gap-1.5 mt-2 flex-wrap">
            <span class="text-[10px] px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 font-semibold">On-time ${onTime}</span>
            <span class="text-[10px] px-2 py-0.5 rounded bg-orange-50 text-orange-700 font-semibold">Late ${late}</span>
            ${unpaid > 0 ? `<span class="text-[10px] px-2 py-0.5 rounded bg-red-50 text-red-700 font-semibold">Unpaid ${unpaid}</span>` : ""}
        </div>`;
}


function renderDelinquencyRows(container, items) {
    if (!Array.isArray(items) || items.length === 0) {
        container.innerHTML = `
            <div class="py-14 text-center text-slate-400 font-bold">
                <div class="flex flex-col items-center gap-2">
                    <span class="material-symbols-outlined text-4xl text-slate-300">analytics</span>
                    <span>Chưa có dữ liệu dự báo. Vui lòng import dữ liệu thanh toán.</span>
                </div>
            </div>`;
        return;
    }

    container.innerHTML = items.map((item, index) => {
        const riskProbabilityRaw = toSafeNumber(item.probability ?? item.risk_probability, 0);
        const riskProbability = Math.max(0, Math.min(1, riskProbabilityRaw));
        const widthPct = (riskProbability * 100).toFixed(0);
        const riskStyle = getRiskColor(riskProbability);

        const taxCodeRaw = (item.tax_code || "---").toString();
        const taxCode = escapeHtml(taxCodeRaw);
        const companyName = escapeHtml(item.company_name || item.name || "---");
        const cluster = escapeHtml(item.cluster || "---");
        const modelVersion = item.model_version ? `<span class="text-[10px] text-slate-500">${escapeHtml(item.model_version)}</span>` : "";

        const paymentSummary = renderPaymentSummary(item.payment_history_summary);
        const predictionMeta = renderPredictionMeta(item);
        const probabilityBreakdown = renderProbabilityBreakdown(item);
        const topReasons = renderTopReasons(item.top_reasons);

        const penaltyAmount = toSafeNumber(item.payment_history_summary?.total_penalties, 0);
        const penaltyText = penaltyAmount > 0 ? `${formatCurrencyCompact(penaltyAmount)} phạt` : "---";
        const detailHref = taxCodeRaw === "---" ? "" : buildDetailUrl(taxCodeRaw);

        return `
            <article class="px-8 py-5 ${index % 2 === 0 ? "bg-white" : "bg-slate-50/40"} hover:bg-slate-50 transition-colors duration-200">
                <div class="grid grid-cols-12 gap-3 items-start">
                    <div class="col-span-12 md:col-span-2">
                        <p class="text-[10px] uppercase tracking-wider text-slate-400 font-bold">MST</p>
                        <p class="font-mono text-sm text-slate-600 mt-1">${taxCode}</p>
                    </div>
                    <div class="col-span-12 md:col-span-3">
                        <h4 class="font-bold text-primary-container text-[15px] leading-tight">${companyName}</h4>
                        ${paymentSummary}
                    </div>
                    <div class="col-span-12 md:col-span-2">
                        <p class="text-[10px] uppercase tracking-wider text-slate-400 font-bold">Nhóm rủi ro</p>
                        <p class="text-sm font-semibold text-slate-700 mt-1">${cluster}</p>
                        ${modelVersion}
                        ${predictionMeta}
                    </div>
                    <div class="col-span-12 md:col-span-3">
                        <p class="text-[10px] uppercase tracking-wider text-slate-400 font-bold">Xác suất tổng</p>
                        <div class="flex items-center gap-2 mt-1.5">
                            <div class="w-24 h-2 bg-slate-100 rounded-full overflow-hidden">
                                <div class="h-full ${riskStyle.bg} rounded-full transition-all duration-500" style="width:${widthPct}%"></div>
                            </div>
                            <span class="px-2 py-0.5 ${riskStyle.text} rounded text-[11px] font-bold min-w-[44px] text-center">${widthPct}%</span>
                        </div>
                        ${probabilityBreakdown}
                        ${topReasons}
                    </div>
                    <div class="col-span-8 md:col-span-1">
                        <p class="text-[10px] uppercase tracking-wider text-slate-400 font-bold">Phạt</p>
                        <p class="font-bold text-primary-container mt-1 text-sm">${escapeHtml(penaltyText)}</p>
                    </div>
                    <div class="col-span-4 md:col-span-1 flex md:justify-end">
                        ${detailHref ? `
                        <a href="${escapeHtml(detailHref)}" class="bg-primary-container text-white px-3 py-1.5 rounded-lg text-xs font-bold hover:opacity-90 transition-all duration-200 inline-flex items-center gap-1">
                            <span class="material-symbols-outlined text-[14px]">visibility</span>
                            Chi tiết
                        </a>
                        ` : `
                        <span class="text-xs text-slate-400 font-semibold">N/A</span>
                        `}
                    </div>
                </div>
            </article>`;
    }).join("");
}


function updateKPIs(data) {
    const predictions = data.predictions || [];
    const total = toSafeNumber(data.total, predictions.length);
    const modelInfo = data.model_info || {};

    const kpi1El = document.querySelector('[data-kpi="high-risk-count"]');
    if (kpi1El) {
        kpi1El.textContent = total.toLocaleString("vi-VN");
    }

    const totalPenalties = predictions.reduce((sum, prediction) => {
        const amount = prediction?.payment_history_summary?.total_penalties;
        return sum + toSafeNumber(amount, 0);
    }, 0);

    const kpi2El = document.querySelector('[data-kpi="estimated-loss"]');
    if (kpi2El) {
        kpi2El.innerHTML = `${formatCurrencyCompact(totalPenalties)} <span class="text-lg">VNĐ</span>`;
    }

    const kpi3El = document.querySelector('[data-kpi="model-accuracy"]');
    if (kpi3El) {
        const source = (modelInfo.source || "unknown").toString();
        const staleCount = toSafeNumber(modelInfo.stale_count, 0);
        const modelVersion = (modelInfo.model_version || "").toString().trim();

        if (source === "ml_model") {
            if (staleCount > 0) {
                kpi3El.textContent = modelVersion ? `ML (${modelVersion}, ${staleCount} stale)` : `ML (${staleCount} stale)`;
            } else {
                kpi3El.textContent = modelVersion ? `ML ${modelVersion}` : "ML Model";
            }
        } else if (source === "statistical_baseline") {
            kpi3El.textContent = modelVersion ? `Baseline ${modelVersion}` : "Baseline";
        } else {
            kpi3El.textContent = "---";
        }
    }

    const paginationInfo = document.querySelector('[data-kpi="pagination-info"]');
    if (paginationInfo) {
        if (total <= 0) {
            paginationInfo.textContent = "Không có kết quả";
            return;
        }

        const start = (state.currentPage - 1) * PAGE_SIZE + 1;
        const end = Math.min(state.currentPage * PAGE_SIZE, total);
        paginationInfo.textContent = `Hiển thị ${start}-${end} của ${total.toLocaleString("vi-VN")} kết quả`;
    }
}


function buildPageButton(page, label, options = {}) {
    const { active = false, disabled = false, icon = false } = options;
    const commonClasses = active
        ? "w-9 h-9 flex items-center justify-center rounded bg-primary-container text-white text-xs font-bold"
        : "w-9 h-9 flex items-center justify-center rounded hover:bg-slate-100 text-xs font-bold text-slate-600 transition-colors";

    return `
        <button
            type="button"
            data-page-target="${page}"
            class="${icon ? "p-1 text-slate-400 hover:bg-slate-100 rounded transition-colors" : commonClasses} ${disabled ? "opacity-30 pointer-events-none" : ""}"
            ${disabled ? "disabled" : ""}
            aria-label="Trang ${page}">
            ${label}
        </button>`;
}


function renderPagination(total, currentPage, pageSize) {
    const container = document.getElementById("delinq-pagination");
    if (!container) return;

    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    state.totalPages = totalPages;

    let html = "";

    html += buildPageButton(
        Math.max(1, currentPage - 1),
        '<span class="material-symbols-outlined text-[18px]">chevron_left</span>',
        { disabled: currentPage <= 1, icon: true }
    );

    const maxVisible = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxVisible / 2));
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);

    if (endPage - startPage < maxVisible - 1) {
        startPage = Math.max(1, endPage - maxVisible + 1);
    }

    if (startPage > 1) {
        html += buildPageButton(1, "1");
        if (startPage > 2) {
            html += '<span class="text-slate-400 mx-1">...</span>';
        }
    }

    for (let page = startPage; page <= endPage; page += 1) {
        html += buildPageButton(page, String(page), { active: page === currentPage });
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            html += '<span class="text-slate-400 mx-1">...</span>';
        }
        html += buildPageButton(totalPages, String(totalPages));
    }

    html += buildPageButton(
        Math.min(totalPages, currentPage + 1),
        '<span class="material-symbols-outlined text-[18px]">chevron_right</span>',
        { disabled: currentPage >= totalPages, icon: true }
    );

    container.innerHTML = html;
}


function attachPaginationHandler() {
    const container = document.getElementById("delinq-pagination");
    if (!container) return;

    container.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-page-target]");
        if (!button || button.disabled) return;

        const pageTarget = Number(button.getAttribute("data-page-target") || "");
        if (!Number.isInteger(pageTarget) || pageTarget < 1 || pageTarget === state.currentPage) {
            return;
        }

        loadDelinquencyData(pageTarget);
    });
}


function setLoadingState(container) {
    container.innerHTML = `
        <div class="py-14 text-center">
            <div class="flex flex-col items-center gap-3">
                <div class="w-8 h-8 border-4 border-primary-container/30 border-t-primary-container rounded-full animate-spin"></div>
                <span class="text-xs text-slate-400 font-bold uppercase tracking-widest">Đang tải dữ liệu...</span>
            </div>
        </div>`;
}


function setErrorState(container, message) {
    container.innerHTML = `
        <div class="py-10 text-center">
            <div class="flex flex-col items-center gap-2">
                <span class="material-symbols-outlined text-3xl text-red-400">error</span>
                <span class="text-error font-bold text-sm">${escapeHtml(message)}</span>
                <button type="button" id="delinq-retry-btn" class="mt-2 px-4 py-2 bg-primary-container text-white rounded-lg text-xs font-bold hover:opacity-90">
                    Thử lại
                </button>
            </div>
        </div>`;

    const retryButton = document.getElementById("delinq-retry-btn");
    if (retryButton) {
        retryButton.addEventListener("click", () => loadDelinquencyData(state.currentPage));
    }
}


function setBatchStatus(message, tone = "info") {
    const element = document.getElementById("delinq-batch-status");
    if (!element) return;

    const toneClasses = {
        info: "bg-blue-50 text-blue-700 border-blue-100",
        success: "bg-emerald-50 text-emerald-700 border-emerald-100",
        warning: "bg-amber-50 text-amber-700 border-amber-100",
        error: "bg-rose-50 text-rose-700 border-rose-100",
    };

    const className = toneClasses[tone] || toneClasses.info;
    element.className = `px-8 py-2 border-b text-xs font-semibold ${className}`;
    element.textContent = message || "";

    if (!message) {
        element.classList.add("hidden");
    } else {
        element.classList.remove("hidden");
    }
}


async function loadDelinquencyData(page = state.currentPage) {
    const listContainer = document.getElementById("delinq-list");
    if (!listContainer) return;

    const requestedPage = Number.isInteger(Number(page)) ? Math.max(1, Number(page)) : 1;
    const requestId = ++latestRequestId;
    setLoadingState(listContainer);

    try {
        const params = new URLSearchParams({
            page: String(requestedPage),
            page_size: String(PAGE_SIZE),
        });

        if (state.freshnessFilter) {
            params.set("freshness", state.freshnessFilter);
        }

        const response = await secureFetch(`${API_BASE}/delinquency?${params.toString()}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        if (requestId !== latestRequestId) {
            return;
        }

        const total = Math.max(0, toSafeNumber(data.total, 0));
        const calculatedTotalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
        const normalizedPage = Math.max(1, Math.min(requestedPage, calculatedTotalPages));

        if (normalizedPage !== requestedPage && total > 0) {
            loadDelinquencyData(normalizedPage);
            return;
        }

        state.currentPage = normalizedPage;
        state.total = total;
        state.totalPages = calculatedTotalPages;

        const normalizedData = {
            ...data,
            total,
            page: normalizedPage,
            page_size: PAGE_SIZE,
        };

        renderDelinquencyRows(listContainer, normalizedData.predictions || []);
        updateKPIs(normalizedData);
        renderPagination(total, state.currentPage, PAGE_SIZE);
        syncUrlState();
    } catch (error) {
        if (requestId !== latestRequestId) {
            return;
        }
        console.error("Delinquency fetch error:", error);
        setErrorState(listContainer, "Lỗi tải dữ liệu danh sách. Vui lòng thử lại.");
    }
}


async function runDelinquencyBatchPredict() {
    const button = document.getElementById("delinq-batch-predict-btn");
    if (!button) return;

    button.disabled = true;
    button.classList.add("opacity-70", "pointer-events-none");
    setBatchStatus("Đang chạy batch predict Delinquency...", "info");

    try {
        const response = await secureFetch(`${API_BASE}/delinquency/predict-batch`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                limit: 300,
                refresh_existing: false,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();
        const failed = toSafeNumber(result.failed, 0);

        setBatchStatus(
            `Batch xong: processed=${toSafeNumber(result.processed, 0)}, created=${toSafeNumber(result.created, 0)}, updated=${toSafeNumber(result.updated, 0)}, failed=${failed}`,
            failed > 0 ? "warning" : "success"
        );

        await loadDelinquencyData(state.currentPage);
    } catch (error) {
        console.error("Batch predict error:", error);
        setBatchStatus("Batch predict thất bại. Vui lòng thử lại.", "error");
    } finally {
        button.disabled = false;
        button.classList.remove("opacity-70", "pointer-events-none");
    }
}


document.addEventListener("DOMContentLoaded", () => {
    const initial = parseInitialStateFromUrl();
    state.currentPage = initial.page;
    state.freshnessFilter = initial.freshness;

    const freshnessSelect = document.getElementById("delinq-freshness-filter");
    if (freshnessSelect) {
        freshnessSelect.value = state.freshnessFilter;
        freshnessSelect.addEventListener("change", (event) => {
            state.freshnessFilter = normalizeFreshnessValue(event.target.value);
            state.currentPage = 1;
            loadDelinquencyData(1);
        });
    }

    const batchButton = document.getElementById("delinq-batch-predict-btn");
    if (batchButton) {
        batchButton.addEventListener("click", runDelinquencyBatchPredict);
    }

    attachPaginationHandler();
    loadDelinquencyData(state.currentPage);
});
