const ALLOWED_FRESHNESS = new Set(["fresh", "aging", "stale", "unknown"]);


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


function normalizeFreshness(value) {
    const normalized = (value || "").toString().trim().toLowerCase();
    return ALLOWED_FRESHNESS.has(normalized) ? normalized : "";
}


function formatCurrencyCompact(value) {
    const num = toSafeNumber(value, 0);
    if (num >= 1e9) return `${(num / 1e9).toFixed(1)} Tỷ VNĐ`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)} Tr VNĐ`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(0)} K VNĐ`;
    return `${num.toLocaleString("vi-VN")} VNĐ`;
}


function getRiskStyle(probability) {
    if (probability > 0.8) return { label: "Rất cao", badge: "bg-red-100 text-red-700" };
    if (probability > 0.6) return { label: "Cao", badge: "bg-orange-100 text-orange-700" };
    if (probability > 0.4) return { label: "Trung bình", badge: "bg-yellow-100 text-yellow-700" };
    if (probability > 0.2) return { label: "Thấp", badge: "bg-blue-100 text-blue-700" };
    return { label: "Rất thấp", badge: "bg-emerald-100 text-emerald-700" };
}


function sourceLabel(scoreSource) {
    const source = (scoreSource || "unknown").toString().toLowerCase();
    if (source === "ml_model") return "ML model";
    if (source === "statistical_baseline") return "Statistical baseline";
    if (source === "no_data") return "No data";
    return "Fallback";
}


function setText(id, value) {
    const element = document.getElementById(id);
    if (!element) return;
    element.textContent = value;
}


function setStatus(message, tone = "info") {
    const statusElement = document.getElementById("detail-status");
    if (!statusElement) return;

    const toneClass = {
        info: "text-slate-500",
        success: "text-emerald-700",
        warning: "text-amber-700",
        error: "text-red-700",
    };

    statusElement.className = `mt-6 text-sm ${toneClass[tone] || toneClass.info}`;
    statusElement.textContent = message;
}


function parsePageState() {
    const params = new URLSearchParams(window.location.search);
    const taxCode = (params.get("tax_code") || "").toString().trim();
    const rawPage = Number(params.get("page") || "1");
    const page = Number.isInteger(rawPage) && rawPage > 0 ? rawPage : 1;
    const freshness = normalizeFreshness(params.get("freshness"));

    return { taxCode, page, freshness };
}


function buildBackUrl(page, freshness) {
    const params = new URLSearchParams();
    if (page > 1) {
        params.set("page", String(page));
    }
    if (freshness) {
        params.set("freshness", freshness);
    }

    const query = params.toString();
    return `delinquency.html${query ? `?${query}` : ""}`;
}


function renderMetaBadges(detail) {
    const holder = document.getElementById("detail-meta-badges");
    if (!holder) return;

    const freshness = (detail.freshness || "unknown").toString().toLowerCase();
    const freshnessClass = freshness === "fresh"
        ? "bg-emerald-50 text-emerald-700 border border-emerald-100"
        : freshness === "aging"
            ? "bg-amber-50 text-amber-700 border border-amber-100"
            : freshness === "stale"
                ? "bg-rose-50 text-rose-700 border border-rose-100"
                : "bg-slate-100 text-slate-600 border border-slate-200";

    const ageDays = Number.isFinite(Number(detail.prediction_age_days)) ? Number(detail.prediction_age_days) : null;
    const ageText = ageDays === null ? "Age: N/A" : `Age: ${ageDays} ngày`;

    holder.innerHTML = `
        <span class="text-[11px] px-2.5 py-1 rounded-full font-bold ${freshnessClass}">${escapeHtml(freshness || "unknown")}</span>
        <span class="text-[11px] px-2.5 py-1 rounded-full font-bold bg-blue-50 text-blue-700 border border-blue-100">${escapeHtml(sourceLabel(detail.score_source))}</span>
        <span class="text-[11px] px-2.5 py-1 rounded-full font-semibold bg-white/15 text-white border border-white/20">${escapeHtml(ageText)}</span>
        ${detail.monotonic_adjusted ? '<span class="text-[11px] px-2.5 py-1 rounded-full font-semibold bg-white/15 text-white border border-white/20">Đã monotonic adjust</span>' : ""}
    `;
}


function renderReasons(detail) {
    const reasonList = document.getElementById("detail-top-reasons");
    if (!reasonList) return;

    const reasons = Array.isArray(detail.top_reasons) ? detail.top_reasons : [];
    if (reasons.length === 0) {
        reasonList.innerHTML = '<li class="text-slate-400">Không có nguyên nhân chi tiết cho doanh nghiệp này.</li>';
        return;
    }

    reasonList.innerHTML = reasons.map((item) => {
        const reason = escapeHtml(item.reason || "Không rõ");
        const weight = Math.max(0, Math.min(1, toSafeNumber(item.weight, 0)));
        const widthPct = (weight * 100).toFixed(0);

        return `
            <li class="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
                <div class="flex items-center justify-between gap-3">
                    <span class="font-medium text-slate-700">${reason}</span>
                    <span class="text-xs font-bold text-slate-500">${widthPct}%</span>
                </div>
                <div class="w-full h-1.5 bg-slate-200 rounded-full mt-2 overflow-hidden">
                    <div class="h-full bg-primary-container rounded-full" style="width:${widthPct}%"></div>
                </div>
            </li>`;
    }).join("");
}


function renderDetail(detail) {
    const probability = Math.max(0, Math.min(1, toSafeNumber(detail.probability, 0)));
    const prob30 = Math.max(0, Math.min(1, toSafeNumber(detail.prob_30d, 0)));
    const prob60 = Math.max(0, Math.min(1, toSafeNumber(detail.prob_60d, 0)));
    const prob90 = Math.max(0, Math.min(1, toSafeNumber(detail.prob_90d, 0)));

    const riskStyle = getRiskStyle(probability);

    setText("detail-company-name", detail.company_name || "Không rõ doanh nghiệp");
    setText("detail-tax-code", detail.tax_code || "---");
    setText("detail-overall-prob", `${(probability * 100).toFixed(0)}%`);

    const riskBadge = document.getElementById("detail-risk-badge");
    if (riskBadge) {
        riskBadge.className = `px-3 py-1 rounded-full text-xs font-bold ${riskStyle.badge}`;
        riskBadge.textContent = riskStyle.label;
    }

    setText("detail-prob-30", `${(prob30 * 100).toFixed(0)}%`);
    setText("detail-prob-60", `${(prob60 * 100).toFixed(0)}%`);
    setText("detail-prob-90", `${(prob90 * 100).toFixed(0)}%`);

    renderReasons(detail);
    renderMetaBadges(detail);

    const summary = detail.payment_history_summary || {};
    setText("detail-payment-on-time", String(toSafeNumber(summary.on_time_count, 0)));
    setText("detail-payment-late", String(toSafeNumber(summary.late_count, 0)));
    setText("detail-payment-unpaid", String(toSafeNumber(summary.unpaid_count, 0)));
    setText("detail-payment-avg-late", String(toSafeNumber(summary.avg_days_late, 0)));

    const totalPenalty = toSafeNumber(summary.total_penalties, 0);
    setText("detail-total-penalty", formatCurrencyCompact(totalPenalty));

    setText("detail-cluster", detail.cluster || "---");
    setText("detail-model-version", detail.model_version || "---");
    setText("detail-prediction-date", detail.prediction_date || "---");

    const ageDays = Number.isFinite(Number(detail.prediction_age_days)) ? Number(detail.prediction_age_days) : null;
    setText("detail-age-days", ageDays === null ? "N/A" : `${ageDays} ngày`);
    setText("detail-score-source", sourceLabel(detail.score_source));

    setStatus("Đã tải hồ sơ doanh nghiệp thành công.", "success");
}


async function fetchDelinquencyDetail(taxCode) {
    const response = await secureFetch(`${API_BASE}/delinquency/${encodeURIComponent(taxCode)}`);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
}


document.addEventListener("DOMContentLoaded", async () => {
    const pageState = parsePageState();

    const backLink = document.getElementById("delinq-back-link");
    if (backLink) {
        backLink.href = buildBackUrl(pageState.page, pageState.freshness);
    }

    if (!pageState.taxCode) {
        setStatus("Thiếu mã số thuế (tax_code) trong đường dẫn.", "error");
        setText("detail-company-name", "Không xác định doanh nghiệp");
        return;
    }

    setStatus("Đang tải hồ sơ doanh nghiệp...", "info");

    try {
        const detail = await fetchDelinquencyDetail(pageState.taxCode);
        renderDetail(detail);
    } catch (error) {
        console.error("Delinquency detail fetch error:", error);
        setStatus("Không thể tải chi tiết doanh nghiệp. Vui lòng thử lại.", "error");
        setText("detail-company-name", "Lỗi tải dữ liệu");
        setText("detail-tax-code", pageState.taxCode);
    }
});
