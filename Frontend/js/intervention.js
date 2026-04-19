(function () {
    const ENDPOINT = `${API_BASE}/monitoring/intervention_effectiveness`;

    const state = {
        windowDays: 90,
        topK: 50,
        latestPayload: null,
        source: "",
        focus: "",
        taxCode: "",
        focusApplied: false,
    };

    const NAV_SOURCE_LABELS = {
        dashboard: "Bảng điều khiển",
        fraud: "Phân tích gian lận",
        delinquency: "Dự báo trễ hạn",
        "delinquency-detail": "Chi tiết trễ hạn",
    };

    const NAV_FOCUS_LABELS = {
        overview: "Tổng quan KPI",
        actions: "Bảng hành động can thiệp",
    };

    const NAV_FOCUS_SECTION_IDS = {
        overview: "intervention-overview-section",
        actions: "intervention-actions-section",
    };

    const ACTION_LABELS = {
        monitor: "THEO DÕI",
        field_audit: "KIỂM TRA THỰC ĐỊA",
        escalated_enforcement: "CƯỠNG CHẾ TĂNG CƯỜNG",
        auto_reminder: "NHẮC HẠN TỰ ĐỘNG",
        structured_outreach: "TIẾP CẬN CÓ CẤU TRÚC",
        targeted_audit: "THANH TRA MỤC TIÊU",
        unassigned: "CHƯA GÁN",
    };

    function byId(id) {
        return document.getElementById(id);
    }

    function clampPercent(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return 0;
        return Math.max(0, Math.min(1, num));
    }

    function formatNumber(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return "--";
        return num.toLocaleString("vi-VN");
    }

    function formatPercent(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return "--";
        return `${(num * 100).toFixed(1)}%`;
    }

    function formatCurrency(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return "--";
        return new Intl.NumberFormat("vi-VN", {
            style: "currency",
            currency: "VND",
            maximumFractionDigits: 0,
        }).format(num);
    }

    function formatTimestamp(value) {
        if (!value) return "--";
        const dt = new Date(value);
        if (Number.isNaN(dt.getTime())) return "--";
        return dt.toLocaleString("vi-VN", { hour12: false });
    }

    function normalizeActionLabel(raw) {
        const base = String(raw || "unassigned").trim().toLowerCase();
        if (!base) return ACTION_LABELS.unassigned;
        if (ACTION_LABELS[base]) return ACTION_LABELS[base];
        return base.replaceAll("_", " ").toUpperCase();
    }

    function escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }

    function normalizeNavigationSource(rawSource) {
        const normalized = String(rawSource || "").trim().toLowerCase();
        return NAV_SOURCE_LABELS[normalized] ? normalized : "";
    }

    function normalizeNavigationFocus(rawFocus) {
        const normalized = String(rawFocus || "").trim().toLowerCase();
        return NAV_FOCUS_SECTION_IDS[normalized] ? normalized : "";
    }

    function renderNavigationHint() {
        const hint = byId("intervention-navigation-hint");
        if (!hint) return;

        if (!state.source && !state.focus && !state.taxCode) {
            hint.classList.add("hidden");
            hint.textContent = "";
            return;
        }

        const sourceText = state.source ? `Mở từ ${NAV_SOURCE_LABELS[state.source]}` : "Mở từ điều hướng nội bộ";
        const focusText = state.focus ? ` • Tập trung: ${NAV_FOCUS_LABELS[state.focus]}` : "";
        const taxCodeText = state.taxCode ? ` • MST: ${state.taxCode}` : "";

        hint.textContent = `${sourceText}${focusText}${taxCodeText}`;
        hint.classList.remove("hidden");
    }

    function focusRequestedSection() {
        if (!state.focus || state.focusApplied) return;
        const targetId = NAV_FOCUS_SECTION_IDS[state.focus];
        const target = byId(targetId);
        if (!target) return;

        state.focusApplied = true;
        window.requestAnimationFrame(() => {
            target.scrollIntoView({ behavior: "smooth", block: "start" });
        });
    }

    function setStatus(type, title, message) {
        const banner = byId("intervention-status-banner");
        const dot = byId("intervention-status-dot");
        const titleEl = byId("intervention-status-text");
        const messageEl = byId("intervention-status-message");

        if (!banner || !dot || !titleEl || !messageEl) return;

        const toneMap = {
            loading: {
                bannerClass: "status-banner rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs",
                dotClass: "inline-block w-2.5 h-2.5 rounded-full bg-slate-400",
            },
            ready: {
                bannerClass: "status-banner rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs",
                dotClass: "inline-block w-2.5 h-2.5 rounded-full bg-emerald-500",
            },
            warning: {
                bannerClass: "status-banner rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs",
                dotClass: "inline-block w-2.5 h-2.5 rounded-full bg-amber-500",
            },
            error: {
                bannerClass: "status-banner rounded-xl border border-red-200 bg-red-50 px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs",
                dotClass: "inline-block w-2.5 h-2.5 rounded-full bg-red-500",
            },
        };

        const tone = toneMap[type] || toneMap.loading;
        banner.className = tone.bannerClass;
        dot.className = tone.dotClass;
        titleEl.textContent = title;
        messageEl.textContent = message;
    }

    function mapDataRealityStatusLabel(rawStatus) {
        const normalized = String(rawStatus || "").toLowerCase();
        if (normalized === "ready") return "DỮ LIỆU THẬT: ĐẠT";
        if (normalized === "warning") return "DỮ LIỆU THẬT: CẦN BỔ SUNG";
        if (normalized === "blocked") return "DỮ LIỆU THẬT: KHÓA";
        if (normalized === "error") return "DỮ LIỆU THẬT: LỖI KIỂM TRA";
        return "DỮ LIỆU THẬT: CHƯA ĐÁNH GIÁ";
    }

    function summarizeDataReality(dataReality) {
        if (!dataReality || typeof dataReality !== "object") {
            return "Chưa có thông tin xác thực dữ liệu train thật.";
        }

        const metrics = dataReality.metrics || {};
        const parts = [];

        const assessmentRatio = Number(metrics.real_assessment_ratio);
        const labelRatio = Number(metrics.real_label_ratio);
        const manualRatio = Number(metrics.manual_label_ratio);

        if (Number.isFinite(assessmentRatio)) {
            parts.push(`assessment thật ${formatPercent(assessmentRatio)}`);
        }
        if (Number.isFinite(labelRatio)) {
            parts.push(`label thật ${formatPercent(labelRatio)}`);
        }
        if (Number.isFinite(manualRatio)) {
            parts.push(`nhãn thực địa ${formatPercent(manualRatio)}`);
        }

        return parts.length
            ? parts.join(" • ")
            : "Chưa có đủ chỉ số xác thực dữ liệu thật.";
    }

    function setText(id, value) {
        const el = byId(id);
        if (!el) return;
        el.textContent = value;
    }

    function renderActionTable(rows) {
        const body = byId("intervention-action-table-body");
        const countEl = byId("intervention-actions-count");
        if (!body) return;

        const safeRows = Array.isArray(rows) ? rows : [];
        if (countEl) {
            countEl.textContent = `${safeRows.length} hành động`;
        }

        if (!safeRows.length) {
            body.innerHTML = "<tr><td colspan=\"8\" class=\"px-4 py-6 text-center text-slate-400 italic\">Không có dữ liệu hành động trong phạm vi lọc.</td></tr>";
            return;
        }

        body.innerHTML = safeRows.map((item) => {
            const action = normalizeActionLabel(item.intervention_action);
            return `
<tr class="hover:bg-surface-container-low transition-colors">
<td class="px-4 py-3 font-semibold text-primary-container">${escapeHtml(action)}</td>
<td class="px-4 py-3">${formatNumber(item.sample_count)}</td>
<td class="px-4 py-3">${formatNumber(item.positive_count)}</td>
<td class="px-4 py-3 font-bold">${formatPercent(item.precision)}</td>
<td class="px-4 py-3">${formatCurrency(item.expected_uplift)}</td>
<td class="px-4 py-3 text-emerald-700 font-semibold">${formatCurrency(item.actual_recovered)}</td>
<td class="px-4 py-3">${formatPercent(item.uplift_realization_ratio)}</td>
<td class="px-4 py-3">${formatCurrency(item.avg_net_recovery)}</td>
</tr>`;
        }).join("");
    }

    function renderActionBars(rows) {
        const container = byId("intervention-action-bars");
        if (!container) return;

        const safeRows = Array.isArray(rows) ? rows : [];
        if (!safeRows.length) {
            container.innerHTML = "<p class=\"text-slate-400 italic text-sm\">Không có dữ liệu để biểu diễn thanh hiệu quả.</p>";
            return;
        }

        container.innerHTML = safeRows.slice(0, 8).map((item) => {
            const label = normalizeActionLabel(item.intervention_action);
            const precision = clampPercent(item.precision);
            return `
<div class="rounded-lg border border-outline-variant/20 bg-surface-container-low p-3">
<div class="flex items-center justify-between gap-2 mb-2">
<p class="text-[11px] font-bold tracking-wide text-slate-600">${escapeHtml(label)}</p>
<span class="text-[11px] font-black text-primary-container">${formatPercent(item.precision)}</span>
</div>
<div class="w-full h-2 rounded-full bg-white overflow-hidden">
<div class="h-full rounded-full bg-primary-container action-meter-fill" style="width:${(precision * 100).toFixed(1)}%"></div>
</div>
<div class="mt-2 flex justify-between text-[10px] text-slate-500 uppercase tracking-wider">
<span>Mẫu ${formatNumber(item.sample_count)}</span>
<span>Thu hồi ròng ${formatCurrency(item.avg_net_recovery)}</span>
</div>
</div>`;
        }).join("");
    }

    function renderPayload(payload) {
        state.latestPayload = payload;

        const schemaReady = Boolean(payload?.schema_ready);
        const dataReality = payload?.data_reality;
        if (!schemaReady) {
            const reason = dataReality && Array.isArray(dataReality.reasons) && dataReality.reasons.length
                ? dataReality.reasons[0]
                : null;
            setStatus(
                "warning",
                "Schema KPI chưa sẵn sàng",
                reason || payload?.message || "Thiếu cột nghiệp vụ trong inspector_labels. Hãy chạy migration mới.",
            );

            [
                "intervention-total-labels",
                "intervention-attempted-labels",
                "intervention-terminal-labels",
                "intervention-positive-recovery",
                "intervention-precision",
                "intervention-precision-sample",
                "intervention-roi-positive",
                "intervention-fn-high-risk",
                "intervention-net-recovery",
                "intervention-avg-recovery",
                "intervention-uplift-ratio",
                "intervention-recovery-ratio",
                "intervention-total-predicted",
                "intervention-total-expected",
                "intervention-total-actual",
                "intervention-total-audit",
            ].forEach((id) => setText(id, "--"));

            setText("intervention-attempted-ratio", "Tỷ lệ: --");
            renderActionTable([]);
            renderActionBars([]);
            setText("intervention-updated-at", `Cập nhật lần cuối: ${new Date().toLocaleTimeString("vi-VN")}`);
            setText("intervention-generated-at", "Thời điểm tạo: --");
            return;
        }

        const summary = payload?.summary || {};
        const metrics = payload?.metrics || {};
        const actions = Array.isArray(payload?.by_intervention_action) ? payload.by_intervention_action : [];
        const baseMessage = `Cửa sổ ${payload.window_days || state.windowDays} ngày • Top-K ${payload.top_k || state.topK}`;

        if (dataReality && !dataReality.ready_for_real_ops) {
            const reasons = Array.isArray(dataReality.reasons) ? dataReality.reasons : [];
            const primaryReason = reasons[0] || "Dữ liệu train thật chưa đạt chuẩn vận hành.";
            setStatus(
                dataReality.hard_ready ? "warning" : "error",
                dataReality.hard_ready ? "KPI có dữ liệu nhưng chưa đạt chuẩn train thật" : "Khóa vận hành KPI train thật",
                `${primaryReason} • ${summarizeDataReality(dataReality)}`,
            );
        } else {
            const label = mapDataRealityStatusLabel(dataReality?.status);
            setStatus(
                "ready",
                "Đồng bộ KPI thành công",
                `${baseMessage} • ${label} • ${summarizeDataReality(dataReality)}`,
            );
        }

        setText("intervention-total-labels", formatNumber(summary.total_labels));
        setText("intervention-attempted-labels", formatNumber(summary.attempted_labels));
        setText(
            "intervention-attempted-ratio",
            `Tỷ lệ: ${formatPercent(Number(summary.attempted_labels || 0) / Math.max(1, Number(summary.total_labels || 0)))}`,
        );
        setText("intervention-terminal-labels", formatNumber(summary.terminal_labels));
        setText("intervention-positive-recovery", `Thu hồi dương: ${formatNumber(summary.positive_recovery_labels)}`);
        setText("intervention-precision", formatPercent(metrics.precision_at_top_k));
        setText("intervention-precision-sample", `Mẫu: ${formatNumber(metrics.precision_top_k_sample)}`);
        setText("intervention-roi-positive", formatPercent(metrics.roi_positive_rate));
        setText("intervention-fn-high-risk", formatPercent(metrics.false_negative_rate_high_risk));
        setText("intervention-net-recovery", formatCurrency(metrics.net_recovery));
        setText("intervention-avg-recovery", formatCurrency(metrics.average_recovery_per_label));

        setText("intervention-uplift-ratio", formatPercent(metrics.expected_vs_actual_uplift_ratio));
        setText("intervention-recovery-ratio", formatPercent(metrics.expected_vs_actual_recovery_ratio));
        setText("intervention-total-predicted", formatCurrency(metrics.total_predicted_uplift));
        setText("intervention-total-expected", formatCurrency(metrics.total_expected_recovery));
        setText("intervention-total-actual", formatCurrency(metrics.total_actual_recovered));
        setText("intervention-total-audit", formatCurrency(metrics.total_audit_cost));

        renderActionTable(actions);
        renderActionBars(actions);

        setText("intervention-updated-at", `Cập nhật lần cuối: ${new Date().toLocaleTimeString("vi-VN")}`);
        setText("intervention-generated-at", `Thời điểm tạo: ${formatTimestamp(payload.generated_at)}`);
    }

    function renderError(message) {
        setStatus(
            "error",
            "Không thể tải KPI can thiệp",
            message || "Không thể kết nối API monitoring/intervention_effectiveness.",
        );
        renderActionTable([]);
        renderActionBars([]);
    }

    function setLoading(loading) {
        const refreshBtn = byId("intervention-refresh-btn");
        if (refreshBtn) {
            refreshBtn.disabled = loading;
            refreshBtn.classList.toggle("opacity-70", loading);
            refreshBtn.classList.toggle("cursor-wait", loading);
        }
    }

    function syncStateToUrl() {
        const params = new URLSearchParams(window.location.search);
        params.set("window_days", String(state.windowDays));
        params.set("top_k", String(state.topK));
        const nextUrl = `${window.location.pathname}?${params.toString()}`;
        window.history.replaceState({}, "", nextUrl);
    }

    function readStateFromUrl() {
        const params = new URLSearchParams(window.location.search);
        const windowDays = Number(params.get("window_days"));
        const topK = Number(params.get("top_k"));
        const source = normalizeNavigationSource(params.get("source"));
        const focus = normalizeNavigationFocus(params.get("focus"));
        const taxCode = String(params.get("tax_code") || "").trim().slice(0, 32);

        if (Number.isFinite(windowDays) && windowDays >= 7 && windowDays <= 365) {
            state.windowDays = Math.round(windowDays);
        }
        if (Number.isFinite(topK) && topK >= 10 && topK <= 500) {
            state.topK = Math.round(topK);
        }

        state.source = source;
        state.focus = focus;
        state.taxCode = taxCode;
        state.focusApplied = false;

        const windowSelect = byId("intervention-window-days");
        const topKSelect = byId("intervention-top-k");
        if (windowSelect) windowSelect.value = String(state.windowDays);
        if (topKSelect) topKSelect.value = String(state.topK);
    }

    async function fetchIntervention() {
        setLoading(true);
        setStatus("loading", "Đang tải dữ liệu KPI", "Đang đồng bộ API intervention_effectiveness...");

        try {
            syncStateToUrl();
            const query = new URLSearchParams({
                window_days: String(state.windowDays),
                top_k: String(state.topK),
            });
            const response = await secureFetch(`${ENDPOINT}?${query.toString()}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const payload = await response.json();
            renderPayload(payload);
            renderNavigationHint();
            focusRequestedSection();
        } catch (error) {
            console.error("Intervention effectiveness fetch failed:", error);
            renderError("Không thể tải dữ liệu từ backend. Hãy kiểm tra API hoặc phiên đăng nhập.");
        } finally {
            setLoading(false);
        }
    }

    function bindEvents() {
        const windowSelect = byId("intervention-window-days");
        const topKSelect = byId("intervention-top-k");
        const refreshBtn = byId("intervention-refresh-btn");

        if (windowSelect) {
            windowSelect.addEventListener("change", () => {
                const value = Number(windowSelect.value);
                state.windowDays = Number.isFinite(value) ? value : state.windowDays;
                fetchIntervention();
            });
        }

        if (topKSelect) {
            topKSelect.addEventListener("change", () => {
                const value = Number(topKSelect.value);
                state.topK = Number.isFinite(value) ? value : state.topK;
                fetchIntervention();
            });
        }

        if (refreshBtn) {
            refreshBtn.addEventListener("click", () => {
                fetchIntervention();
            });
        }
    }

    document.addEventListener("DOMContentLoaded", () => {
        readStateFromUrl();
        renderNavigationHint();
        bindEvents();
        fetchIntervention();
    });
})();
