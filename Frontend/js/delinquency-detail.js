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
