(function () {
    const AUDIT_ENDPOINT = `${API_BASE}/monitoring/audit_value_effectiveness`;
    const VAT_ENDPOINT = `${API_BASE}/monitoring/vat_refund_effectiveness`;
    const ROLLOUT_ENDPOINT = `${API_BASE}/monitoring/specialized_rollout_status?include_split_snapshot=true`;

    const state = {
        windowDays: 90,
        topK: 50,
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
        audit: "Tuyến Audit Value",
        vat: "Tuyến VAT Refund",
        rollout: "Quyết định triển khai",
        split: "Ảnh chụp split-trigger",
    };

    const NAV_FOCUS_SECTION_IDS = {
        audit: "specialized-track-audit",
        vat: "specialized-track-vat",
        rollout: "specialized-rollout-decision-section",
        split: "specialized-split-section",
    };

    const DISPLAY_LABELS = {
        unassigned: "CHƯA GÁN",
        monitor: "THEO DÕI",
        field_audit: "KIỂM TRA THỰC ĐỊA",
        escalated_enforcement: "CƯỠNG CHẾ TĂNG CƯỜNG",
        auto_reminder: "NHẮC HẠN TỰ ĐỘNG",
        structured_outreach: "TIẾP CẬN CÓ CẤU TRÚC",
        targeted_audit: "THANH TRA MỤC TIÊU",
    };

    function byId(id) {
        return document.getElementById(id);
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

    function escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }

    function normalizeLabel(raw) {
        const normalized = String(raw || "unassigned").trim().toLowerCase();
        if (DISPLAY_LABELS[normalized]) {
            return DISPLAY_LABELS[normalized];
        }
        return normalized.replaceAll("_", " ").toUpperCase();
    }

    function formatBoolean(value) {
        return value ? "có" : "không";
    }

    function mapReadyLabel(ready) {
        return ready ? "SẴN SÀNG" : "BỊ CHẶN";
    }

    function mapRolloutStatusLabel(rawStatus) {
        const normalized = String(rawStatus || "").toLowerCase();
        if (normalized === "ready_for_phase_d") return "SẴN SÀNG PHA D";
        if (normalized === "review_required") return "RÀ SOÁT DỮ LIỆU THẬT";
        if (normalized === "conditional_go") return "CHẤP THUẬN CÓ ĐIỀU KIỆN";
        if (normalized === "no_go") return "KHÔNG CHẤP THUẬN";
        if (normalized === "insufficient_artifacts") return "THIẾU MINH CHỨNG";
        if (!normalized || normalized === "unavailable") return "CHƯA CÓ DỮ LIỆU";
        return normalized.replaceAll("_", " ").toUpperCase();
    }

    function mapDecisionStatusLabel(rawStatus) {
        const normalized = String(rawStatus || "").toLowerCase();
        if (normalized === "go") return "PHÊ DUYỆT";
        if (normalized === "conditional_go") return "PHÊ DUYỆT CÓ ĐIỀU KIỆN";
        if (normalized === "no_go") return "KHÔNG PHÊ DUYỆT";
        if (!normalized || normalized === "unavailable") return "CHƯA CÓ DỮ LIỆU";
        return normalized.replaceAll("_", " ").toUpperCase();
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
            return "Chưa có thông tin kiểm định dữ liệu thật.";
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
            : "Chưa có đủ chỉ số để xác nhận dữ liệu train thật.";
    }

    function collectDataRealityActions(actions, dataReality) {
        const baseActions = Array.isArray(actions) ? actions.map((item) => String(item)) : [];
        const seen = new Set(baseActions);

        if (dataReality && Array.isArray(dataReality.reasons)) {
            dataReality.reasons.slice(0, 3).forEach((reason) => {
                const item = `Gate dữ liệu thật: ${String(reason)}`;
                if (!seen.has(item)) {
                    seen.add(item);
                    baseActions.unshift(item);
                }
            });
        }

        return baseActions;
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
        const hint = byId("specialized-navigation-hint");
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

    function setText(id, value) {
        const el = byId(id);
        if (!el) return;
        el.textContent = value;
    }

    function setRolloutBanner(type, title, message) {
        const banner = byId("specialized-rollout-banner");
        const dot = byId("specialized-rollout-dot");
        const titleEl = byId("specialized-rollout-text");
        const messageEl = byId("specialized-rollout-message");

        if (!banner || !dot || !titleEl || !messageEl) return;

        const toneMap = {
            loading: {
                bannerClass: "rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs",
                dotClass: "inline-block w-2.5 h-2.5 rounded-full bg-slate-400",
            },
            success: {
                bannerClass: "rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs",
                dotClass: "inline-block w-2.5 h-2.5 rounded-full bg-emerald-500",
            },
            warning: {
                bannerClass: "rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs",
                dotClass: "inline-block w-2.5 h-2.5 rounded-full bg-amber-500",
            },
            error: {
                bannerClass: "rounded-xl border border-red-200 bg-red-50 px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs",
                dotClass: "inline-block w-2.5 h-2.5 rounded-full bg-red-500",
            },
        };

        const tone = toneMap[type] || toneMap.loading;
        banner.className = tone.bannerClass;
        dot.className = tone.dotClass;
        titleEl.textContent = title;
        messageEl.textContent = message;
    }

    function setSchemaStatus(prefix, schemaReady, message) {
        const badge = byId(`${prefix}-schema-badge`);
        const note = byId(`${prefix}-schema-note`);

        if (badge) {
            if (schemaReady) {
                badge.className = "status-pill text-[10px] uppercase tracking-widest font-bold px-2 py-1 rounded bg-emerald-50 text-emerald-700";
                badge.textContent = "SCHEMA SẴN SÀNG";
            } else {
                badge.className = "status-pill text-[10px] uppercase tracking-widest font-bold px-2 py-1 rounded bg-amber-50 text-amber-700";
                badge.textContent = "THIẾU SCHEMA";
            }
        }

        if (note) {
            note.textContent = schemaReady
                ? "Dữ liệu tuyến đã sẵn sàng để đánh giá hiệu quả."
                : (message || "Thiếu cột nghiệp vụ, cần migration/bổ sung dữ liệu.");
        }
    }

    function renderAudit(payload) {
        const body = byId("audit-lane-table-body");

        if (!payload || !payload.schema_ready) {
            setSchemaStatus("audit", false, payload?.message);
            [
                "audit-total-labels",
                "audit-precision",
                "audit-precision-sample",
                "audit-recovered-rate",
                "audit-roi-rate",
                "audit-realization",
                "audit-net-recovery",
            ].forEach((id) => setText(id, "--"));
            if (body) {
                body.innerHTML = "<tr><td colspan=\"6\" class=\"px-3 py-4 text-center text-slate-400 italic\">Tuyến Audit Value chưa sẵn sàng schema.</td></tr>";
            }
            return;
        }

        setSchemaStatus("audit", true, "");

        const auditDataReality = payload?.data_reality;
        if (auditDataReality && !auditDataReality.ready_for_real_ops) {
            const badge = byId("audit-schema-badge");
            const note = byId("audit-schema-note");
            if (badge) {
                badge.className = auditDataReality.hard_ready
                    ? "status-pill text-[10px] uppercase tracking-widest font-bold px-2 py-1 rounded bg-amber-50 text-amber-700"
                    : "status-pill text-[10px] uppercase tracking-widest font-bold px-2 py-1 rounded bg-red-50 text-red-700";
                badge.textContent = auditDataReality.hard_ready ? "DATA CẦN BỔ SUNG" : "DATA BỊ CHẶN";
            }
            if (note) {
                const reason = Array.isArray(auditDataReality.reasons) && auditDataReality.reasons.length
                    ? auditDataReality.reasons[0]
                    : "Dữ liệu train thật chưa đạt chuẩn vận hành.";
                note.textContent = String(reason);
            }
        }

        const summary = payload.summary || {};
        const metrics = payload.metrics || {};
        const rows = Array.isArray(payload.by_lane) ? payload.by_lane : [];

        setText("audit-total-labels", formatNumber(summary.total_labels));
        setText("audit-precision", formatPercent(metrics.precision_at_top_k));
        setText("audit-precision-sample", `Mẫu ${formatNumber(metrics.precision_top_k_sample)}`);
        setText("audit-recovered-rate", formatPercent(metrics.recovered_rate));
        setText("audit-roi-rate", formatPercent(metrics.roi_positive_rate));
        setText("audit-realization", formatPercent(metrics.expected_vs_actual_recovery_ratio));
        setText("audit-net-recovery", formatCurrency(metrics.net_recovery));

        if (!body) return;

        if (!rows.length) {
            body.innerHTML = "<tr><td colspan=\"6\" class=\"px-3 py-4 text-center text-slate-400 italic\">Không có luồng Audit Value trong phạm vi lọc.</td></tr>";
            return;
        }

        body.innerHTML = rows.map((row) => `
<tr class="hover:bg-surface-container-low transition-colors">
<td class="px-3 py-2 font-semibold text-primary-container">${escapeHtml(normalizeLabel(row.lane))}</td>
<td class="px-3 py-2">${formatNumber(row.sample_count)}</td>
<td class="px-3 py-2">${formatPercent(row.success_rate)}</td>
<td class="px-3 py-2">${formatCurrency(row.expected_recovery)}</td>
<td class="px-3 py-2 text-emerald-700 font-semibold">${formatCurrency(row.actual_recovered)}</td>
<td class="px-3 py-2">${formatPercent(row.recovery_realization_ratio)}</td>
</tr>
`).join("");
    }

    function renderVat(payload) {
        const body = byId("vat-action-table-body");

        if (!payload || !payload.schema_ready) {
            setSchemaStatus("vat", false, payload?.message);
            [
                "vat-total-labels",
                "vat-precision",
                "vat-precision-sample",
                "vat-recovered-labels",
                "vat-roi-rate",
                "vat-fn-rate",
                "vat-net-recovery",
            ].forEach((id) => setText(id, "--"));
            if (body) {
                body.innerHTML = "<tr><td colspan=\"7\" class=\"px-3 py-4 text-center text-slate-400 italic\">Tuyến VAT Refund chưa sẵn sàng schema.</td></tr>";
            }
            return;
        }

        setSchemaStatus("vat", true, "");

        const vatDataReality = payload?.data_reality;
        if (vatDataReality && !vatDataReality.ready_for_real_ops) {
            const badge = byId("vat-schema-badge");
            const note = byId("vat-schema-note");
            if (badge) {
                badge.className = vatDataReality.hard_ready
                    ? "status-pill text-[10px] uppercase tracking-widest font-bold px-2 py-1 rounded bg-amber-50 text-amber-700"
                    : "status-pill text-[10px] uppercase tracking-widest font-bold px-2 py-1 rounded bg-red-50 text-red-700";
                badge.textContent = vatDataReality.hard_ready ? "DATA CẦN BỔ SUNG" : "DATA BỊ CHẶN";
            }
            if (note) {
                const reason = Array.isArray(vatDataReality.reasons) && vatDataReality.reasons.length
                    ? vatDataReality.reasons[0]
                    : "Dữ liệu train thật chưa đạt chuẩn vận hành.";
                note.textContent = String(reason);
            }
        }

        const summary = payload.summary || {};
        const metrics = payload.metrics || {};
        const rows = Array.isArray(payload.by_intervention_action) ? payload.by_intervention_action : [];

        setText("vat-total-labels", formatNumber(summary.total_labels));
        setText("vat-precision", formatPercent(metrics.precision_at_top_k));
        setText("vat-precision-sample", `Mẫu ${formatNumber(metrics.precision_top_k_sample)}`);
        setText("vat-recovered-labels", formatNumber(summary.recovered_labels));
        setText("vat-roi-rate", formatPercent(metrics.roi_positive_rate));
        setText("vat-fn-rate", formatPercent(metrics.false_negative_rate_high_risk));
        setText("vat-net-recovery", formatCurrency(metrics.net_recovery));

        if (!body) return;

        if (!rows.length) {
            body.innerHTML = "<tr><td colspan=\"7\" class=\"px-3 py-4 text-center text-slate-400 italic\">Không có hành động VAT Refund trong phạm vi lọc.</td></tr>";
            return;
        }

        body.innerHTML = rows.map((row) => `
<tr class="hover:bg-surface-container-low transition-colors">
<td class="px-3 py-2 font-semibold text-primary-container">${escapeHtml(normalizeLabel(row.intervention_action))}</td>
<td class="px-3 py-2">${formatNumber(row.sample_count)}</td>
<td class="px-3 py-2">${formatPercent(row.precision)}</td>
<td class="px-3 py-2">${formatCurrency(row.expected_recovery)}</td>
<td class="px-3 py-2 text-emerald-700 font-semibold">${formatCurrency(row.actual_recovered)}</td>
<td class="px-3 py-2">${formatPercent(row.recovery_realization_ratio)}</td>
<td class="px-3 py-2">${formatCurrency(row.avg_net_recovery)}</td>
</tr>
`).join("");
    }

    function renderSplitSnapshot(splitSnapshot) {
        const container = byId("specialized-split-rules");
        if (!container) return;

        const trackStatus = splitSnapshot?.track_status || {};
        const auditTrack = trackStatus.audit_value;
        const vatTrack = trackStatus.vat_refund;

        const blocks = [];

        [
            { key: "audit_value", title: "Audit Value", payload: auditTrack },
            { key: "vat_refund", title: "VAT Refund", payload: vatTrack },
        ].forEach((track) => {
            if (!track.payload) {
                blocks.push(`
<div class="rounded-lg border border-outline-variant/20 bg-surface-container-low p-3">
<p class="text-[10px] uppercase tracking-widest text-slate-400 font-bold">${track.title}</p>
<p class="mt-1 text-xs text-slate-500">Không có dữ liệu split-trigger.</p>
</div>`);
                return;
            }

            const ready = Boolean(track.payload.ready_for_split);
            const enabledCount = Number(track.payload.enabled_rule_count || 0);
            const blockingCount = Number(track.payload.blocking_rule_count || 0);

            blocks.push(`
<div class="rounded-lg border ${ready ? "border-emerald-200 bg-emerald-50" : "border-amber-200 bg-amber-50"} p-3">
<div class="flex items-center justify-between gap-2">
<p class="text-[10px] uppercase tracking-widest text-slate-500 font-bold">${track.title}</p>
        <span class="text-[10px] font-black ${ready ? "text-emerald-700" : "text-amber-700"}">${mapReadyLabel(ready)}</span>
</div>
        <p class="mt-1 text-xs text-slate-600">Quy tắc bật: ${formatNumber(enabledCount)} • Quy tắc chặn: ${formatNumber(blockingCount)}</p>
</div>`);
        });

        container.innerHTML = blocks.join("");
    }

    function rolloutBadge(status) {
        const normalized = String(status || "").toLowerCase();
        if (normalized === "ready_for_phase_d") {
            return {
                badge: "bg-emerald-50 text-emerald-700",
                tone: "success",
                title: "Sẵn sàng Pha D",
            };
        }
        if (normalized === "review_required") {
            return {
                badge: "bg-amber-50 text-amber-700",
                tone: "warning",
                title: "Rà soát dữ liệu thật",
            };
        }
        if (normalized === "conditional_go") {
            return {
                badge: "bg-amber-50 text-amber-700",
                tone: "warning",
                title: "Chấp thuận có điều kiện",
            };
        }
        if (normalized === "no_go") {
            return {
                badge: "bg-red-50 text-red-700",
                tone: "error",
                title: "Không chấp thuận",
            };
        }
        if (normalized === "insufficient_artifacts") {
            return {
                badge: "bg-slate-100 text-slate-600",
                tone: "warning",
                title: "Thiếu minh chứng",
            };
        }
        return {
            badge: "bg-blue-50 text-blue-700",
            tone: "warning",
            title: "Cần rà soát",
        };
    }

    function renderRollout(payload) {
        const status = rolloutBadge(payload?.rollout_status);
        const dataReality = payload?.data_reality;
        const badge = byId("specialized-decision-badge");
        if (badge) {
            badge.className = `status-pill text-[10px] uppercase tracking-widest font-bold px-2 py-1 rounded ${status.badge}`;
            badge.textContent = mapRolloutStatusLabel(payload?.rollout_status);
        }

        const splitSnapshot = payload?.split_trigger_snapshot || {};
        const trackStatus = splitSnapshot?.track_status || {};
        const auditReady = Boolean(trackStatus?.audit_value?.ready_for_split);
        const vatReady = Boolean(trackStatus?.vat_refund?.ready_for_split);

        setText("specialized-readiness-score", `${Number(splitSnapshot?.readiness_score || 0).toFixed(1)}%`);
        setText("specialized-audit-ready", mapReadyLabel(auditReady));
        setText("specialized-vat-ready", mapReadyLabel(vatReady));

        const actions = collectDataRealityActions(payload?.recommended_actions, dataReality);
        setText("specialized-actions-count", formatNumber(actions.length));

        const decision = payload?.artifacts?.go_no_go || {};
        const decisionStatus = mapDecisionStatusLabel(decision.decision_status);
        const dataRealityStatus = mapDataRealityStatusLabel(dataReality?.status);
        const decisionMessage = String(decision.message || "Chưa có thông điệp từ báo cáo go/no-go.");

        setText("specialized-decision", `${decisionStatus} • Pha D=${decision.go_live_phase_d ? "CÓ" : "KHÔNG"} • ${dataRealityStatus}`);
        setText("specialized-decision-message", decisionMessage);

        const actionsList = byId("specialized-actions-list");
        if (actionsList) {
            if (!actions.length) {
                actionsList.innerHTML = "<li class=\"text-slate-400 italic\">Không có hành động đề xuất tại thời điểm hiện tại.</li>";
            } else {
                actionsList.innerHTML = actions.map((item) => `
<li class="rounded-lg border border-outline-variant/20 bg-surface-container-low px-3 py-2">${escapeHtml(item)}</li>
`).join("");
            }
        }

        const availability = payload?.availability || {};
        const artifactSummaryParts = [
            `chất lượng audit=${formatBoolean(availability.audit_quality)}`,
            `chất lượng vat=${formatBoolean(availability.vat_quality)}`,
            `báo cáo pilot=${formatBoolean(availability.pilot_report)}`,
            `báo cáo go/no-go=${formatBoolean(availability.go_no_go_report)}`,
        ];

        if (dataReality && typeof dataReality === "object") {
            const metrics = dataReality.metrics || {};
            if (Number.isFinite(Number(metrics.real_assessment_ratio))) {
                artifactSummaryParts.push(`assessment thật=${formatPercent(metrics.real_assessment_ratio)}`);
            }
            if (Number.isFinite(Number(metrics.real_label_ratio))) {
                artifactSummaryParts.push(`label thật=${formatPercent(metrics.real_label_ratio)}`);
            }
        }

        const artifactSummary = artifactSummaryParts.join(" • ");

        setText("specialized-artifact-summary", `Tệp minh chứng: ${artifactSummary}`);

        const rolloutSummary = `Trạng thái triển khai: ${mapRolloutStatusLabel(payload?.rollout_status)} • Ứng viên Pha D: ${formatBoolean(payload?.phase_d_candidate)}`;
        const realitySummary = summarizeDataReality(dataReality);
        const reasons = dataReality && Array.isArray(dataReality.reasons) ? dataReality.reasons : [];

        let bannerTone = status.tone;
        let bannerTitle = status.title;
        let bannerMessage = `${rolloutSummary} • ${realitySummary}`;

        if (dataReality && !dataReality.ready_for_real_ops) {
            bannerTone = dataReality.hard_ready ? "warning" : "error";
            bannerTitle = dataReality.hard_ready ? "Cần bổ sung dữ liệu train thật" : "Khóa vận hành bằng dữ liệu thật";
            bannerMessage = `${reasons[0] || "Dữ liệu train thật chưa đạt chuẩn."} • ${realitySummary}`;
        }

        setRolloutBanner(
            bannerTone,
            bannerTitle,
            bannerMessage,
        );

        renderSplitSnapshot(splitSnapshot);
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

        const windowSelect = byId("specialized-window-days");
        const topKSelect = byId("specialized-top-k");
        if (windowSelect) windowSelect.value = String(state.windowDays);
        if (topKSelect) topKSelect.value = String(state.topK);
    }

    async function fetchJson(url, label) {
        const response = await secureFetch(url);
        if (!response.ok) {
            throw new Error(`${label} HTTP ${response.status}`);
        }
        return response.json();
    }

    function setLoading(loading) {
        const refreshBtn = byId("specialized-refresh-btn");
        if (!refreshBtn) return;
        refreshBtn.disabled = loading;
        refreshBtn.classList.toggle("opacity-70", loading);
        refreshBtn.classList.toggle("cursor-wait", loading);
    }

    async function fetchAll() {
        setLoading(true);
        setRolloutBanner("loading", "Đang đồng bộ dữ liệu", "Đang tải đồng thời Audit Value, VAT Refund và trạng thái triển khai...");

        try {
            syncStateToUrl();
            const query = new URLSearchParams({
                window_days: String(state.windowDays),
                top_k: String(state.topK),
            }).toString();

            const [auditPayload, vatPayload, rolloutPayload] = await Promise.all([
                fetchJson(`${AUDIT_ENDPOINT}?${query}`, "audit_value_effectiveness"),
                fetchJson(`${VAT_ENDPOINT}?${query}`, "vat_refund_effectiveness"),
                fetchJson(ROLLOUT_ENDPOINT, "specialized_rollout_status"),
            ]);

            renderAudit(auditPayload);
            renderVat(vatPayload);
            renderRollout(rolloutPayload);

            const latestGeneratedAt = [
                auditPayload?.generated_at,
                vatPayload?.generated_at,
                rolloutPayload?.generated_at,
            ].filter(Boolean).sort().reverse()[0];

            setText("specialized-updated-at", `Cập nhật lần cuối: ${new Date().toLocaleTimeString("vi-VN")}`);
            if (latestGeneratedAt) {
                setText("specialized-updated-at", `Cập nhật lần cuối: ${formatTimestamp(latestGeneratedAt)}`);
            }

            renderNavigationHint();
            focusRequestedSection();
        } catch (error) {
            console.error("Specialized workspace fetch failed:", error);
            setRolloutBanner(
                "error",
                "Không thể tải chuyên đề thanh tra",
                "Ít nhất một API monitoring không phản hồi. Kiểm tra API hoặc phiên đăng nhập.",
            );
        } finally {
            setLoading(false);
        }
    }

    function bindEvents() {
        const windowSelect = byId("specialized-window-days");
        const topKSelect = byId("specialized-top-k");
        const refreshBtn = byId("specialized-refresh-btn");

        if (windowSelect) {
            windowSelect.addEventListener("change", () => {
                const value = Number(windowSelect.value);
                state.windowDays = Number.isFinite(value) ? value : state.windowDays;
                fetchAll();
            });
        }

        if (topKSelect) {
            topKSelect.addEventListener("change", () => {
                const value = Number(topKSelect.value);
                state.topK = Number.isFinite(value) ? value : state.topK;
                fetchAll();
            });
        }

        if (refreshBtn) {
            refreshBtn.addEventListener("click", () => {
                fetchAll();
            });
        }
    }

    document.addEventListener("DOMContentLoaded", () => {
        readStateFromUrl();
        renderNavigationHint();
        bindEvents();
        fetchAll();
    });
})();
