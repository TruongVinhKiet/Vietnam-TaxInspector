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


function formatSplitGateTimestamp(raw) {
    if (!raw) return "--";
    const dt = new Date(raw);
    if (Number.isNaN(dt.getTime())) return "--";
    return dt.toLocaleString("vi-VN", { hour12: false });
}


function normalizeSplitTriggerStatus(rawStatus) {
    const payload = rawStatus && typeof rawStatus === "object" ? rawStatus : {};
    const hasPayload = Object.keys(payload).length > 0;

    if (!hasPayload) {
        return {
            ready: false,
            schemaReady: false,
            readinessScore: 0,
            enabledRules: 0,
            passedRules: 0,
            blockedTracks: [],
            reason: "Chưa tải được trạng thái split-trigger cho hồ sơ doanh nghiệp này.",
            generatedAt: "",
        };
    }

    const schemaReady = payload.schema_ready === true;
    const ready = schemaReady && payload.ready === true;

    const readinessRaw = Number(payload.readiness_score);
    const readinessScore = Number.isFinite(readinessRaw)
        ? Math.max(0, Math.min(100, readinessRaw))
        : 0;

    const totals = payload.totals && typeof payload.totals === "object" ? payload.totals : {};
    const enabledRules = Number.isFinite(Number(totals.enabled_rules)) ? Number(totals.enabled_rules) : 0;
    const passedRules = Number.isFinite(Number(totals.passed_rules)) ? Number(totals.passed_rules) : 0;

    const trackStatus = payload.track_status && typeof payload.track_status === "object" ? payload.track_status : {};
    const blockedTracks = Object.entries(trackStatus)
        .filter(([, value]) => !(value && value.ready_for_split))
        .map(([trackName, value]) => {
            const blockingRuleCount = Number(value && value.blocking_rule_count ? value.blocking_rule_count : 0);
            const normalizedTrack = String(trackName || "").replaceAll("_", " ").toUpperCase();
            return blockingRuleCount > 0
                ? `${normalizedTrack} (${blockingRuleCount} chặn)`
                : normalizedTrack;
        });

    let reason = String(payload.reason || "").trim();
    if (!reason) {
        if (!schemaReady) {
            reason = "Schema KPI chưa sẵn sàng cho quản trị split-trigger.";
        } else if (ready) {
            reason = "Split-trigger đã sẵn sàng cho luồng can thiệp.";
        } else {
            reason = "Split-trigger chưa đạt KPI, tạm giữ can thiệp ở chế độ theo dõi.";
        }
    }

    return {
        ready,
        schemaReady,
        readinessScore,
        enabledRules,
        passedRules,
        blockedTracks,
        reason,
        generatedAt: payload.generated_at || "",
    };
}


function renderSplitTriggerGate(rawStatus) {
    const gate = normalizeSplitTriggerStatus(rawStatus);

    const badge = document.getElementById("detail-split-gate-badge");
    const summary = document.getElementById("detail-split-gate-summary");
    const updated = document.getElementById("detail-split-gate-updated");

    let badgeLabel = "SẴN SÀNG";
    let badgeClass = "bg-emerald-100 text-emerald-700";
    if (!gate.schemaReady) {
        badgeLabel = "LƯỢC ĐỒ";
        badgeClass = "bg-slate-200 text-slate-700";
    } else if (!gate.ready) {
        badgeLabel = "ĐANG KHÓA";
        badgeClass = "bg-amber-100 text-amber-700";
    }

    if (badge) {
        badge.className = `inline-flex items-center px-3 py-1 rounded-full text-xs font-bold ${badgeClass}`;
        badge.textContent = badgeLabel;
    }
    if (summary) {
        const core = `Mức sẵn sàng ${gate.readinessScore.toFixed(1)}% • Quy tắc ${gate.passedRules}/${gate.enabledRules}`;
        const blocked = gate.blockedTracks.length ? ` • Đang chặn: ${gate.blockedTracks.join(", ")}` : "";
        summary.textContent = gate.ready ? `${core}. ${gate.reason}` : `${core}${blocked}. ${gate.reason}`;
    }
    if (updated) {
        updated.textContent = `Cập nhật: ${formatSplitGateTimestamp(gate.generatedAt)}`;
    }

    return gate;
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
    if (source === "ml_model") return "Mô hình ML";
    if (source === "statistical_baseline") return "Đường cơ sở thống kê";
    if (source === "no_data") return "Không có dữ liệu";
    return "Dự phòng";
}


function normalizeEarlyWarning(rawWarning) {
    const payload = rawWarning && typeof rawWarning === "object" ? rawWarning : {};
    const queueRaw = String(payload.queue || "monitor").toLowerCase();
    const levelRaw = String(payload.level || "low").toLowerCase();

    const queue = queueRaw === "priority_review" || queueRaw === "watchlist" ? queueRaw : "monitor";
    const level = ["critical", "high", "medium", "low"].includes(levelRaw) ? levelRaw : "low";
    const hasWarning = Boolean(payload.has_warning) || queue !== "monitor" || level !== "low";
    const reason = String(payload.reason || "").trim();
    const tags = Array.isArray(payload.tags) ? payload.tags.map((tag) => String(tag || "").trim()).filter(Boolean).slice(0, 5) : [];

    return {
        hasWarning,
        queue,
        level,
        reason,
        tags,
    };
}


function warningQueueLabel(queue) {
    if (queue === "priority_review") return "Ưu tiên rà soát";
    if (queue === "watchlist") return "Danh sách theo dõi";
    return "Theo dõi";
}


function warningLevelLabel(level) {
    if (level === "critical") return "NGHIÊM TRỌNG";
    if (level === "high") return "CAO";
    if (level === "medium") return "TRUNG BÌNH";
    return "THẤP";
}


function interventionActionLabel(action) {
    if (action === "escalated_enforcement") return "Cưỡng chế tăng cường";
    if (action === "field_audit") return "Kiểm tra thực địa";
    if (action === "structured_outreach") return "Tiếp cận có cấu trúc";
    if (action === "auto_reminder") return "Nhắc hạn tự động";
    return "Theo dõi";
}


function interventionActionClass(action) {
    if (action === "escalated_enforcement") return "bg-rose-100 text-rose-700";
    if (action === "field_audit") return "bg-orange-100 text-orange-700";
    if (action === "structured_outreach") return "bg-amber-100 text-amber-700";
    if (action === "auto_reminder") return "bg-blue-100 text-blue-700";
    return "bg-slate-100 text-slate-700";
}


function getDefaultInterventionSteps(action) {
    if (action === "escalated_enforcement") {
        return [
            "Kích hoạt quy trình cưỡng chế theo quy định nội bộ.",
            "Phối hợp đội thu nợ để giảm tồn đọng trong 30 ngày.",
        ];
    }
    if (action === "field_audit") {
        return [
            "Mở soát xét hồ sơ chi tiết theo chu kỳ gần nhất.",
            "Phân công cán bộ xử lý theo nhóm rủi ro cao.",
        ];
    }
    if (action === "structured_outreach") {
        return [
            "Liên hệ doanh nghiệp theo playbook delinquency.",
            "Đặt mốc cam kết nộp bổ sung trong 5 ngày làm việc.",
        ];
    }
    if (action === "auto_reminder") {
        return [
            "Gửi nhắc hạn tự động trước 7 ngày.",
            "Theo dõi phản hồi trong 72 giờ và cập nhật trạng thái.",
        ];
    }
    return [
        "Theo dõi thêm 1-2 chu kỳ nộp thuế gần nhất.",
        "Duy trì nhắc lịch nộp tờ khai qua kênh tự động.",
    ];
}


function normalizeInterventionUplift(rawIntervention, detail) {
    const payload = rawIntervention && typeof rawIntervention === "object" ? rawIntervention : {};
    const probability = Math.max(0, Math.min(1, toSafeNumber(detail.probability, 0)));
    const actionRaw = String(payload.recommended_action || "").trim().toLowerCase();
    const supportedActions = [
        "monitor",
        "auto_reminder",
        "structured_outreach",
        "field_audit",
        "escalated_enforcement",
    ];

    const fallbackAction = probability >= 0.75
        ? "field_audit"
        : probability >= 0.4
            ? "structured_outreach"
            : probability >= 0.2
                ? "auto_reminder"
                : "monitor";

    const action = supportedActions.includes(actionRaw) ? actionRaw : fallbackAction;
    const priorityScoreRaw = toSafeNumber(payload.priority_score, Math.round(probability * 70));
    const priorityScore = Math.max(0, Math.min(100, Math.round(priorityScoreRaw)));

    const expectedRiskReduction = Math.max(0, toSafeNumber(payload.expected_risk_reduction_pp, probability * 12));
    const expectedPenaltySaving = Math.max(0, toSafeNumber(payload.expected_penalty_saving, 0));
    const expectedCollectionUplift = Math.max(0, toSafeNumber(payload.expected_collection_uplift, 0));

    const confidenceRaw = String(payload.confidence || "").trim().toLowerCase();
    const confidence = ["low", "medium", "high"].includes(confidenceRaw) ? confidenceRaw : "medium";

    const rationale = String(payload.rationale || "").trim()
        || `Khuyến nghị ${interventionActionLabel(action)} theo mức rủi ro hiện tại ${(probability * 100).toFixed(0)}%.`;

    const nextSteps = Array.isArray(payload.next_steps)
        ? payload.next_steps.map((step) => String(step || "").trim()).filter(Boolean).slice(0, 4)
        : [];

    return {
        action,
        priorityScore,
        expectedRiskReduction,
        expectedPenaltySaving,
        expectedCollectionUplift,
        confidence,
        rationale,
        nextSteps: nextSteps.length ? nextSteps : getDefaultInterventionSteps(action),
    };
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


function buildDetailInterventionWorkspaceUrl(pageState, taxCode) {
    const params = new URLSearchParams();
    params.set("source", "delinquency-detail");
    params.set("focus", "actions");
    params.set("window_days", "90");
    params.set("top_k", "50");

    const normalizedTaxCode = String(taxCode || "").trim();
    if (normalizedTaxCode) {
        params.set("tax_code", normalizedTaxCode);
    }

    if (pageState?.page > 1) {
        params.set("page", String(pageState.page));
    }
    if (pageState?.freshness) {
        params.set("freshness", pageState.freshness);
    }

    return `intervention.html?${params.toString()}`;
}


function syncDetailWorkspaceLink(pageState, taxCode) {
    const link = document.getElementById("detail-open-intervention-workspace");
    if (!link) return;
    link.href = buildDetailInterventionWorkspaceUrl(pageState, taxCode);
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
    const ageText = ageDays === null ? "Tuổi dự báo: Không có" : `Tuổi dự báo: ${ageDays} ngày`;
    const earlyWarning = normalizeEarlyWarning(detail.early_warning);
    const warningBadgeClass = earlyWarning.queue === "priority_review"
        ? "bg-rose-50 text-rose-700 border border-rose-100"
        : earlyWarning.queue === "watchlist"
            ? "bg-amber-50 text-amber-700 border border-amber-100"
            : "bg-blue-50 text-blue-700 border border-blue-100";

    const freshnessLabel = freshness === "fresh"
        ? "Mới"
        : freshness === "aging"
            ? "Đang cũ"
            : freshness === "stale"
                ? "Cũ"
                : "Không rõ";

    holder.innerHTML = `
        <span class="text-[11px] px-2.5 py-1 rounded-full font-bold ${freshnessClass}">${escapeHtml(freshnessLabel)}</span>
        <span class="text-[11px] px-2.5 py-1 rounded-full font-bold bg-blue-50 text-blue-700 border border-blue-100">${escapeHtml(sourceLabel(detail.score_source))}</span>
        <span class="text-[11px] px-2.5 py-1 rounded-full font-bold ${warningBadgeClass}">${escapeHtml(warningQueueLabel(earlyWarning.queue))}</span>
        <span class="text-[11px] px-2.5 py-1 rounded-full font-semibold bg-white/15 text-white border border-white/20">${escapeHtml(ageText)}</span>
        ${detail.monotonic_adjusted ? '<span class="text-[11px] px-2.5 py-1 rounded-full font-semibold bg-white/15 text-white border border-white/20">Đã hiệu chỉnh đơn điệu</span>' : ""}
    `;
}


function renderEarlyWarningDetail(detail) {
    const warning = normalizeEarlyWarning(detail.early_warning);
    setText("detail-warning-queue", warningQueueLabel(warning.queue));
    setText("detail-warning-level", warningLevelLabel(warning.level));
    setText("detail-warning-reason", warning.reason || "Chưa ghi nhận tín hiệu cảnh báo sớm đáng kể trong kỳ gần đây.");

    const tagsHolder = document.getElementById("detail-warning-tags");
    if (!tagsHolder) return;
    if (!warning.tags.length) {
        tagsHolder.innerHTML = '<span class="text-[10px] text-slate-400 italic">Không có tag bổ sung.</span>';
        return;
    }

    tagsHolder.innerHTML = warning.tags
        .map((tag) => `<span class="text-[10px] px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-semibold">${escapeHtml(tag)}</span>`)
        .join("");
}


function renderInterventionUplift(detail, splitGate = null) {
    const payload = normalizeInterventionUplift(detail.intervention_uplift, detail);
    const gate = splitGate && typeof splitGate === "object" ? splitGate : normalizeSplitTriggerStatus(null);

    if (!gate.ready) {
        payload.action = "monitor";
        payload.priorityScore = Math.min(payload.priorityScore, 35);
        payload.rationale = `${payload.rationale} Split-trigger đang khóa nên chỉ mở luồng theo dõi.`;
        payload.nextSteps = [
            "Giữ hồ sơ ở chế độ theo dõi cho đến khi split-trigger chuyển sang SẴN SÀNG.",
            `Lý do gate: ${gate.reason}`,
        ];
    }

    const actionBadge = document.getElementById("detail-intervention-action");
    if (actionBadge) {
        actionBadge.className = `inline-flex items-center px-3 py-1 rounded-full text-xs font-bold ${interventionActionClass(payload.action)}`;
        actionBadge.textContent = interventionActionLabel(payload.action);
    }

    setText("detail-intervention-priority", String(payload.priorityScore));
    setText("detail-intervention-confidence", payload.confidence.toUpperCase());
    setText("detail-intervention-risk-reduction", `${payload.expectedRiskReduction.toFixed(1)} pp`);
    setText("detail-intervention-penalty-saving", formatCurrencyCompact(payload.expectedPenaltySaving));
    setText("detail-intervention-collection-uplift", formatCurrencyCompact(payload.expectedCollectionUplift));
    setText("detail-intervention-rationale", payload.rationale);

    const nextStepsHolder = document.getElementById("detail-intervention-next-steps");
    if (!nextStepsHolder) return;

    nextStepsHolder.innerHTML = payload.nextSteps
        .map((step) => `<li class="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2 leading-relaxed">${escapeHtml(step)}</li>`)
        .join("");
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
    setText("detail-age-days", ageDays === null ? "Không có" : `${ageDays} ngày`);
    setText("detail-score-source", sourceLabel(detail.score_source));
    renderEarlyWarningDetail(detail);
    const splitGate = renderSplitTriggerGate(detail.split_trigger_status || null);
    renderInterventionUplift(detail, splitGate);

    setStatus("Đã tải hồ sơ doanh nghiệp thành công.", "success");
}


async function fetchDelinquencyDetail(taxCode) {
    const response = await secureFetch(`${API_BASE}/delinquency/${encodeURIComponent(taxCode)}`);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
}

async function fetchPaymentTimeline(taxCode) {
    try {
        const response = await secureFetch(`${API_BASE}/delinquency/${encodeURIComponent(taxCode)}/payment-timeline`);
        if (!response.ok) return null;
        return response.json();
    } catch (e) {
        console.error("[Detail] Timeline fetch error:", e);
        return null;
    }
}

// ────────────────────────────────────────────────────────────
//  CHART COLORS
// ────────────────────────────────────────────────────────────
const DETAIL_COLORS = {
    primary: "#002147", sky: "#0284c7", violet: "#7c3aed",
    emerald: "#10b981", amber: "#f59e0b", orange: "#f97316",
    rose: "#ef4444", slate: "#64748b", slateLight: "#94a3b8",
    slateBg: "#f8fafc", gridLine: "#e2e8f0", white: "#ffffff",
};

const STATUS_COLORS = { on_time: "#10b981", late: "#f97316", unpaid: "#ef4444", partial: "#f59e0b" };
const STATUS_LABELS = { on_time: "Đúng hạn", late: "Trễ hạn", unpaid: "Chưa nộp", partial: "Nộp thiếu" };

// ────────────────────────────────────────────────────────────
//  CHART 1: Payment Timeline
// ────────────────────────────────────────────────────────────
function renderTimelineChart(timelineData) {
    const canvas = document.getElementById("detail-timeline-chart");
    if (!canvas || !timelineData?.timeline?.length) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const W = rect.width;
    const H = 200;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.scale(dpr, dpr);

    const pad = { top: 20, right: 20, bottom: 35, left: 50 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.fillStyle = DETAIL_COLORS.slateBg;
    ctx.fillRect(0, 0, W, H);

    const data = timelineData.timeline;
    const n = data.length;
    if (!n) return;

    // Scale
    const maxAmount = Math.max(...data.map(d => Math.max(d.amount_due, d.amount_paid)), 1);
    const xScale = (i) => pad.left + (i / Math.max(1, n - 1)) * plotW;
    const yScale = (v) => pad.top + plotH - (v / maxAmount) * plotH;

    // Grid
    ctx.strokeStyle = DETAIL_COLORS.gridLine;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (i / 4) * plotH;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    }

    // Amount due line (light)
    ctx.beginPath();
    ctx.strokeStyle = DETAIL_COLORS.slateLight + "88";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    data.forEach((d, i) => {
        const x = xScale(i), y = yScale(d.amount_due);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Amount paid line
    ctx.beginPath();
    ctx.strokeStyle = DETAIL_COLORS.sky;
    ctx.lineWidth = 2;
    data.forEach((d, i) => {
        const x = xScale(i), y = yScale(d.amount_paid);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Status dots
    data.forEach((d, i) => {
        const x = xScale(i);
        const y = yScale(d.amount_paid);
        const color = STATUS_COLORS[d.status] || DETAIL_COLORS.slate;
        ctx.beginPath();
        ctx.arc(x, y, d.days_late > 30 ? 5 : 4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = DETAIL_COLORS.white;
        ctx.lineWidth = 1.5;
        ctx.stroke();
    });

    // X labels (show every few)
    ctx.fillStyle = DETAIL_COLORS.slateLight;
    ctx.font = "9px Inter, sans-serif";
    ctx.textAlign = "center";
    const step = Math.max(1, Math.floor(n / 8));
    data.forEach((d, i) => {
        if (i % step === 0 || i === n - 1) {
            const dateStr = d.due_date ? d.due_date.substring(5) : "";
            ctx.fillText(dateStr, xScale(i), H - pad.bottom + 15);
        }
    });
}

// ────────────────────────────────────────────────────────────
//  CHART 2: Risk Radar (multi-dimensional profile)
// ────────────────────────────────────────────────────────────
function renderDetailRadarChart(detail) {
    const canvas = document.getElementById("detail-radar-chart");
    if (!canvas || !detail) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const size = Math.min(rect.width, 240);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = size + "px";
    canvas.style.height = size + "px";
    ctx.scale(dpr, dpr);

    const cx = size / 2, cy = size / 2, R = size * 0.33;

    const prob = Math.max(0, Math.min(1, toSafeNumber(detail.probability, 0)));
    const p30 = Math.max(0, Math.min(1, toSafeNumber(detail.prob_30d, 0)));
    const p60 = Math.max(0, Math.min(1, toSafeNumber(detail.prob_60d, 0)));
    const p90 = Math.max(0, Math.min(1, toSafeNumber(detail.prob_90d, 0)));
    const summary = detail.payment_history_summary || {};
    const lateRatio = summary.total_periods ? (summary.late_count + summary.unpaid_count) / summary.total_periods : 0;
    const penaltyNorm = Math.min(1, toSafeNumber(summary.total_penalties, 0) / 5000000);

    const axes = [
        { label: "P(30d)", value: p30 },
        { label: "P(60d)", value: p60 },
        { label: "P(90d)", value: p90 },
        { label: "Trễ hạn", value: Math.min(1, lateRatio) },
        { label: "Phạt", value: penaltyNorm },
        { label: "Tổng", value: prob },
    ];
    const n = axes.length;
    const angleStep = (2 * Math.PI) / n;

    ctx.fillStyle = DETAIL_COLORS.slateBg;
    ctx.fillRect(0, 0, size, size);

    // Grid
    for (let r = 0.25; r <= 1; r += 0.25) {
        ctx.beginPath();
        for (let i = 0; i <= n; i++) {
            const angle = -Math.PI / 2 + i * angleStep;
            const x = cx + Math.cos(angle) * R * r;
            const y = cy + Math.sin(angle) * R * r;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.strokeStyle = DETAIL_COLORS.gridLine;
        ctx.lineWidth = 0.5;
        ctx.stroke();
    }

    // Axes + labels
    axes.forEach((ax, i) => {
        const angle = -Math.PI / 2 + i * angleStep;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(angle) * R, cy + Math.sin(angle) * R);
        ctx.strokeStyle = DETAIL_COLORS.gridLine;
        ctx.lineWidth = 0.5;
        ctx.stroke();

        const lx = cx + Math.cos(angle) * (R + 20);
        const ly = cy + Math.sin(angle) * (R + 20);
        ctx.fillStyle = DETAIL_COLORS.primary;
        ctx.font = "bold 9px Inter";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(ax.label, lx, ly);
    });

    // Polygon
    ctx.beginPath();
    axes.forEach((ax, i) => {
        const angle = -Math.PI / 2 + i * angleStep;
        const r = Math.max(0, Math.min(1, ax.value)) * R;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = "rgba(239, 68, 68, 0.15)";
    ctx.fill();
    ctx.strokeStyle = DETAIL_COLORS.rose;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Dots
    axes.forEach((ax, i) => {
        const angle = -Math.PI / 2 + i * angleStep;
        const r = Math.max(0, Math.min(1, ax.value)) * R;
        ctx.beginPath();
        ctx.arc(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r, 3, 0, Math.PI * 2);
        ctx.fillStyle = DETAIL_COLORS.rose;
        ctx.fill();
    });
}

// ────────────────────────────────────────────────────────────
//  CHART 3: Bullet Chart – Probability vs Thresholds
// ────────────────────────────────────────────────────────────
function renderBulletChart(detail) {
    const canvas = document.getElementById("detail-bullet-chart");
    if (!canvas || !detail) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const W = rect.width;
    const H = 240;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.scale(dpr, dpr);

    ctx.fillStyle = DETAIL_COLORS.slateBg;
    ctx.fillRect(0, 0, W, H);

    const pad = { top: 15, right: 25, bottom: 15, left: 80 };
    const plotW = W - pad.left - pad.right;

    const horizons = [
        { label: "30 ngày", value: toSafeNumber(detail.prob_30d, 0) },
        { label: "60 ngày", value: toSafeNumber(detail.prob_60d, 0) },
        { label: "90 ngày", value: toSafeNumber(detail.prob_90d, 0) },
    ];

    const barH = 40;
    const gap = 20;
    const totalH = horizons.length * (barH + gap);
    const startY = pad.top + (H - pad.top - pad.bottom - totalH) / 2;

    const thresholds = [
        { value: 0.2, color: "#10b981", label: "Thấp" },
        { value: 0.4, color: "#f59e0b", label: "TB" },
        { value: 0.6, color: "#f97316", label: "Cao" },
        { value: 0.8, color: "#ef4444", label: "Rất cao" },
    ];

    horizons.forEach((h, i) => {
        const y = startY + i * (barH + gap);

        // Background zones
        let prevX = pad.left;
        thresholds.forEach((t, ti) => {
            const nextVal = ti < thresholds.length - 1 ? thresholds[ti + 1].value : 1.0;
            const x1 = pad.left + t.value * plotW;
            const x2 = pad.left + nextVal * plotW;
            ctx.fillStyle = t.color + "22";
            ctx.fillRect(prevX, y, x2 - prevX, barH);
            prevX = x2;
        });
        // last zone
        ctx.fillStyle = "#ef444422";
        ctx.fillRect(prevX, y, pad.left + plotW - prevX, barH);

        // Actual value bar
        const barW = Math.max(2, h.value * plotW);
        const valueColor = h.value > 0.8 ? DETAIL_COLORS.rose : h.value > 0.6 ? DETAIL_COLORS.orange : h.value > 0.4 ? DETAIL_COLORS.amber : h.value > 0.2 ? DETAIL_COLORS.sky : DETAIL_COLORS.emerald;
        ctx.fillStyle = valueColor;
        ctx.beginPath();
        ctx.roundRect(pad.left, y + barH * 0.25, barW, barH * 0.5, 3);
        ctx.fill();

        // Value text
        ctx.fillStyle = DETAIL_COLORS.primary;
        ctx.font = "bold 11px Inter";
        ctx.textAlign = "left";
        ctx.fillText(`${(h.value * 100).toFixed(0)}%`, pad.left + barW + 6, y + barH / 2 + 4);

        // Label
        ctx.fillStyle = DETAIL_COLORS.primary;
        ctx.font = "11px Inter";
        ctx.textAlign = "right";
        ctx.fillText(h.label, pad.left - 10, y + barH / 2 + 4);

        // Threshold markers
        thresholds.forEach(t => {
            const tx = pad.left + t.value * plotW;
            ctx.strokeStyle = t.color + "88";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(tx, y);
            ctx.lineTo(tx, y + barH);
            ctx.stroke();
        });
    });

    // Threshold labels at bottom
    ctx.font = "8px Inter";
    ctx.textAlign = "center";
    thresholds.forEach(t => {
        const tx = pad.left + t.value * plotW;
        ctx.fillStyle = t.color;
        ctx.fillText(`${(t.value * 100)}%`, tx, startY + totalH + 12);
    });
}

// ────────────────────────────────────────────────────────────
//  CHART 4: Payment Status Donut
// ────────────────────────────────────────────────────────────
function renderDetailDonut(timelineData) {
    const canvas = document.getElementById("detail-donut-chart");
    if (!canvas || !timelineData?.status_counts) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const size = Math.min(rect.width, 220);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = size + "px";
    canvas.style.height = size + "px";
    ctx.scale(dpr, dpr);

    const cx = size / 2, cy = size / 2;
    const R = size * 0.32;

    ctx.fillStyle = DETAIL_COLORS.slateBg;
    ctx.fillRect(0, 0, size, size);

    const counts = timelineData.status_counts;
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    if (!total) return;

    let startAngle = -Math.PI / 2;
    const entries = Object.entries(counts).filter(([, v]) => v > 0);

    entries.forEach(([key, count]) => {
        const sliceAngle = (count / total) * 2 * Math.PI;
        ctx.beginPath();
        ctx.arc(cx, cy, R, startAngle, startAngle + sliceAngle);
        ctx.lineWidth = R * 0.45;
        ctx.strokeStyle = STATUS_COLORS[key] || DETAIL_COLORS.slate;
        ctx.stroke();
        startAngle += sliceAngle;
    });

    // Center
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "bold 20px Inter";
    ctx.fillStyle = DETAIL_COLORS.primary;
    ctx.fillText(total.toString(), cx, cy - 6);
    ctx.font = "9px Inter";
    ctx.fillStyle = DETAIL_COLORS.slate;
    ctx.fillText("Tổng kỳ", cx, cy + 10);

    // Legend below donut
    let legendY = cy + R + 25;
    ctx.font = "9px Inter";
    entries.forEach(([key, count]) => {
        const pct = ((count / total) * 100).toFixed(0);
        ctx.fillStyle = STATUS_COLORS[key] || DETAIL_COLORS.slate;
        ctx.beginPath();
        ctx.arc(cx - 45, legendY, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = DETAIL_COLORS.primary;
        ctx.textAlign = "left";
        ctx.fillText(`${STATUS_LABELS[key] || key}: ${count} (${pct}%)`, cx - 35, legendY + 3);
        legendY += 16;
    });
}

// ────────────────────────────────────────────────────────────
//  CHART 5: Sparkline – Monthly Payment Amounts
// ────────────────────────────────────────────────────────────
function renderSparklineChart(timelineData) {
    const canvas = document.getElementById("detail-sparkline-chart");
    if (!canvas || !timelineData?.monthly_series?.length) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const W = rect.width;
    const H = 180;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.scale(dpr, dpr);

    const pad = { top: 15, right: 20, bottom: 35, left: 60 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.fillStyle = DETAIL_COLORS.slateBg;
    ctx.fillRect(0, 0, W, H);

    const data = timelineData.monthly_series;
    const n = data.length;
    const maxVal = Math.max(...data.map(d => Math.max(d.total_due, d.total_paid)), 1);

    const xs = (i) => pad.left + (i / Math.max(1, n - 1)) * plotW;
    const ys = (v) => pad.top + plotH - (v / maxVal) * plotH;

    // Grid
    ctx.strokeStyle = DETAIL_COLORS.gridLine;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 3; i++) {
        const y = pad.top + (i / 3) * plotH;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    }

    // Due area (light fill)
    ctx.beginPath();
    data.forEach((d, i) => { i === 0 ? ctx.moveTo(xs(i), ys(d.total_due)) : ctx.lineTo(xs(i), ys(d.total_due)); });
    ctx.lineTo(xs(n - 1), ys(0));
    ctx.lineTo(xs(0), ys(0));
    ctx.closePath();
    ctx.fillStyle = "rgba(148,163,184,0.1)";
    ctx.fill();

    // Due line
    ctx.beginPath();
    ctx.strokeStyle = DETAIL_COLORS.slateLight;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    data.forEach((d, i) => { i === 0 ? ctx.moveTo(xs(i), ys(d.total_due)) : ctx.lineTo(xs(i), ys(d.total_due)); });
    ctx.stroke();
    ctx.setLineDash([]);

    // Paid line
    ctx.beginPath();
    const grad = ctx.createLinearGradient(pad.left, 0, W - pad.right, 0);
    grad.addColorStop(0, DETAIL_COLORS.sky);
    grad.addColorStop(1, DETAIL_COLORS.emerald);
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.5;
    data.forEach((d, i) => { i === 0 ? ctx.moveTo(xs(i), ys(d.total_paid)) : ctx.lineTo(xs(i), ys(d.total_paid)); });
    ctx.stroke();

    // Dots
    data.forEach((d, i) => {
        ctx.beginPath();
        ctx.arc(xs(i), ys(d.total_paid), 3, 0, Math.PI * 2);
        ctx.fillStyle = DETAIL_COLORS.sky;
        ctx.fill();
    });

    // X labels
    ctx.fillStyle = DETAIL_COLORS.slateLight;
    ctx.font = "8px Inter";
    ctx.textAlign = "center";
    const labelStep = Math.max(1, Math.floor(n / 6));
    data.forEach((d, i) => {
        if (i % labelStep === 0 || i === n - 1) {
            ctx.fillText(d.month, xs(i), H - 8);
        }
    });
}

// ────────────────────────────────────────────────────────────
//  CHART 6: Feature Importance (Horizontal Bar)
// ────────────────────────────────────────────────────────────
function renderFeatureChart(detail) {
    const canvas = document.getElementById("detail-feature-chart");
    if (!canvas || !detail?.top_reasons?.length) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const W = rect.width;
    const H = 180;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.scale(dpr, dpr);

    ctx.fillStyle = DETAIL_COLORS.slateBg;
    ctx.fillRect(0, 0, W, H);

    const pad = { top: 10, right: 20, bottom: 10, left: 130 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const reasons = detail.top_reasons.slice(0, 5);
    const maxWeight = Math.max(...reasons.map(r => Math.abs(r.weight || 0)), 0.1);
    const barH = Math.min(26, plotH / reasons.length - 6);

    const barColors = [DETAIL_COLORS.rose, DETAIL_COLORS.orange, DETAIL_COLORS.amber, DETAIL_COLORS.sky, DETAIL_COLORS.violet];

    reasons.forEach((r, i) => {
        const y = pad.top + (i / reasons.length) * plotH + (plotH / reasons.length - barH) / 2;
        const w = (Math.abs(r.weight || 0) / maxWeight) * plotW;
        const color = barColors[i % barColors.length];

        // Bar
        ctx.fillStyle = color + "CC";
        ctx.beginPath();
        ctx.roundRect(pad.left, y, Math.max(4, w), barH, 4);
        ctx.fill();

        // Percentage
        ctx.fillStyle = color;
        ctx.font = "bold 10px Inter";
        ctx.textAlign = "left";
        ctx.fillText(`${((r.weight || 0) * 100).toFixed(0)}%`, pad.left + w + 6, y + barH / 2 + 3);

        // Label
        const label = (r.reason || "").length > 18 ? (r.reason || "").substring(0, 17) + "…" : (r.reason || "");
        ctx.fillStyle = DETAIL_COLORS.primary;
        ctx.font = "10px Inter";
        ctx.textAlign = "right";
        ctx.fillText(label, pad.left - 8, y + barH / 2 + 3);
    });
}


// ── DOMContentLoaded ───────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
    const pageState = parsePageState();

    const backLink = document.getElementById("delinq-back-link");
    if (backLink) {
        backLink.href = buildBackUrl(pageState.page, pageState.freshness);
    }

    syncDetailWorkspaceLink(pageState, pageState.taxCode);

    if (!pageState.taxCode) {
        setStatus("Thiếu mã số thuế (tax_code) trong đường dẫn.", "error");
        setText("detail-company-name", "Không xác định doanh nghiệp");
        return;
    }

    setStatus("Đang tải hồ sơ doanh nghiệp...", "info");

    try {
        const [detail, timelineData] = await Promise.all([
            fetchDelinquencyDetail(pageState.taxCode),
            fetchPaymentTimeline(pageState.taxCode),
        ]);

        syncDetailWorkspaceLink(pageState, detail?.tax_code || pageState.taxCode);
        renderDetail(detail);

        // Render advanced charts
        renderDetailRadarChart(detail);
        renderBulletChart(detail);
        renderFeatureChart(detail);

        if (timelineData) {
            renderTimelineChart(timelineData);
            renderDetailDonut(timelineData);
            renderSparklineChart(timelineData);
        }
    } catch (error) {
        console.error("Delinquency detail fetch error:", error);
        setStatus("Không thể tải chi tiết doanh nghiệp. Vui lòng thử lại.", "error");
        setText("detail-company-name", "Lỗi tải dữ liệu");
        setText("detail-tax-code", pageState.taxCode);
    }
});

