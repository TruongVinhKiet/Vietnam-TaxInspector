(function () {
    const POLL_INTERVAL_MS = 60000;
    const HEALTH_ENDPOINT = `${API_BASE}/delinquency/health/cache?fresh_days=7&stale_days=30`;
    const KPI_SPLIT_ENDPOINT = `${API_BASE}/monitoring/split_trigger_status`;
    const KPI_SNAPSHOTS_ENDPOINT = `${API_BASE}/monitoring/kpi_snapshots?days=30&limit=200`;
    const KPI_ALERTS_ENDPOINT = `${API_BASE}/monitoring/split_trigger_alerts?days=14&min_pass_rate=0.70&min_recent_pass_rate=0.65&min_drift_pp=0.08&min_track_pass_rate=0.65&stale_snapshot_hours=12`;
    const SPECIALIZED_ROLLOUT_ENDPOINT = `${API_BASE}/monitoring/specialized_rollout_status?include_split_snapshot=true`;
    const KPI_PERSIST_STORAGE_KEY = "dashboard_kpi_persist_snapshot_v1";
    const KPI_WATCH_ALERT_CODES = [
        {
            code: "snapshot_stale",
            description: "Snapshot KPI vượt ngưỡng độ mới SLA.",
            normalMessage: "Snapshot vẫn nằm trong SLA độ mới.",
        },
        {
            code: "readiness_drift_down",
            description: "Readiness score giảm mạnh so với giai đoạn trước.",
            normalMessage: "Readiness drift đang ổn định.",
        },
        {
            code: "track_pass_rate_low",
            description: "Tỷ lệ đạt theo track thấp hơn ngưỡng vận hành.",
            normalMessage: "Tỷ lệ đạt theo track đang trong ngưỡng.",
        },
    ];

    let pollTimer = null;

    function byId(id) {
        return document.getElementById(id);
    }

    function toPercent(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return "N/A";
        return `${(num * 100).toFixed(1)}%`;
    }

    function toSignedPercent(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return "--";
        const sign = num > 0 ? "+" : "";
        return `${sign}${(num * 100).toFixed(1)}%`;
    }

    function toSignedPoints(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return "--";
        const sign = num > 0 ? "+" : "";
        return `${sign}${num.toFixed(1)} pt`;
    }

    function formatTimestamp(raw) {
        if (!raw) return "--";
        const dt = new Date(raw);
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

    function statusLabel(status) {
        const normalized = (status || "").toString().toLowerCase();
        if (normalized === "healthy") return "ỔN ĐỊNH";
        if (normalized === "warning") return "CẢNH BÁO";
        if (normalized === "critical") return "NGHIÊM TRỌNG";
        if (normalized === "no_data") return "THIẾU DỮ LIỆU";
        return "LỖI";
    }

    function badgeClasses(status) {
        const normalized = (status || "").toString().toLowerCase();
        if (normalized === "healthy") return "text-emerald-700 bg-emerald-50";
        if (normalized === "warning") return "text-amber-700 bg-amber-50";
        if (normalized === "critical") return "text-red-700 bg-red-50";
        if (normalized === "no_data") return "text-slate-600 bg-slate-100";
        return "text-red-700 bg-red-50";
    }

    function progressClasses(status) {
        const normalized = (status || "").toString().toLowerCase();
        if (normalized === "healthy") return "bg-emerald-500";
        if (normalized === "warning") return "bg-amber-500";
        if (normalized === "critical") return "bg-red-500";
        return "bg-slate-400";
    }

    function setBadge(status) {
        const el = byId("dashboard-delinquency-health-badge");
        if (!el) return;
        el.className = `text-[10px] font-bold px-2 py-0.5 rounded ${badgeClasses(status)}`;
        el.textContent = statusLabel(status);
    }

    function setProgress(status, coverageRatio) {
        const el = byId("dashboard-delinquency-health-progress");
        if (!el) return;

        const coverage = Number(coverageRatio);
        const width = Number.isFinite(coverage) ? Math.max(0, Math.min(100, coverage * 100)) : 0;
        el.style.width = `${width.toFixed(1)}%`;
        el.className = `h-full transition-all duration-500 ${progressClasses(status)}`;
    }

    function updateHealthWidget(payload) {
        const status = (payload?.status || "error").toString().toLowerCase();
        const coverageRatio = payload?.coverage?.coverage_ratio;
        const staleRatio = payload?.freshness?.ratios?.stale;
        const alertCount = Array.isArray(payload?.alerts) ? payload.alerts.length : 0;
        const topAlert = Array.isArray(payload?.alerts) && payload.alerts.length > 0 ? payload.alerts[0] : null;

        setBadge(status);
        setProgress(status, coverageRatio);

        const statusEl = byId("dashboard-delinquency-health-status");
        if (statusEl) statusEl.textContent = statusLabel(status);

        const subtextEl = byId("dashboard-delinquency-health-subtext");
        if (subtextEl) {
            const coverageText = toPercent(coverageRatio);
            const staleText = toPercent(staleRatio);
            subtextEl.textContent = `Độ phủ ${coverageText}, Cũ ${staleText}, Cảnh báo ${alertCount}`;
        }

        const coverageEl = byId("dashboard-delinquency-health-coverage");
        if (coverageEl) coverageEl.textContent = toPercent(coverageRatio);

        const staleEl = byId("dashboard-delinquency-health-stale");
        if (staleEl) staleEl.textContent = toPercent(staleRatio);

        const alertEl = byId("dashboard-delinquency-health-alert");
        if (alertEl) {
            if (topAlert?.message) {
                alertEl.textContent = topAlert.message;
                if (topAlert.severity === "critical") {
                    alertEl.className = "mt-3 text-[11px] text-red-600 min-h-[2.75rem]";
                } else {
                    alertEl.className = "mt-3 text-[11px] text-amber-700 min-h-[2.75rem]";
                }
            } else {
                alertEl.textContent = "Không có cảnh báo về dữ liệu cũ/độ phủ tại thời điểm hiện tại.";
                alertEl.className = "mt-3 text-[11px] text-emerald-700 min-h-[2.75rem]";
            }
        }

        const updatedEl = byId("dashboard-delinquency-health-updated");
        if (updatedEl) {
            updatedEl.textContent = `Cập nhật: ${new Date().toLocaleTimeString("vi-VN")}`;
        }
    }

    function setWidgetError(message) {
        setBadge("error");
        setProgress("error", 0);

        const statusEl = byId("dashboard-delinquency-health-status");
        if (statusEl) statusEl.textContent = "ERROR";

        const subtextEl = byId("dashboard-delinquency-health-subtext");
        if (subtextEl) subtextEl.textContent = "Không thể tải trạng thái sức khỏe từ backend.";

        const alertEl = byId("dashboard-delinquency-health-alert");
        if (alertEl) {
            alertEl.className = "mt-3 text-[11px] text-red-600 min-h-[2.75rem]";
            alertEl.textContent = message || "Vui lòng kiểm tra API /api/delinquency/health/cache.";
        }

        const updatedEl = byId("dashboard-delinquency-health-updated");
        if (updatedEl) updatedEl.textContent = `Lỗi lúc: ${new Date().toLocaleTimeString("vi-VN")}`;
    }

    async function fetchAndRenderHealth() {
        try {
            const response = await secureFetch(HEALTH_ENDPOINT);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const payload = await response.json();
            updateHealthWidget(payload);
        } catch (error) {
            console.error("Dashboard delinquency health error:", error);
            setWidgetError("Không thể kết nối endpoint health.");
        }
    }

    function rolloutBadgeClass(status) {
        if (status === "ready_for_phase_d") return "bg-emerald-50 text-emerald-700";
        if (status === "conditional_go") return "bg-amber-50 text-amber-700";
        if (status === "no_go") return "bg-red-50 text-red-700";
        if (status === "insufficient_artifacts") return "bg-slate-100 text-slate-600";
        if (status === "review_required") return "bg-blue-50 text-blue-700";
        return "bg-red-50 text-red-700";
    }

    function rolloutBadgeLabel(status) {
        if (status === "ready_for_phase_d") return "SẴN SÀNG PHA D";
        if (status === "conditional_go") return "CHẤP THUẬN CÓ ĐIỀU KIỆN";
        if (status === "no_go") return "KHÔNG CHẤP THUẬN";
        if (status === "insufficient_artifacts") return "THIẾU TỆP";
        if (status === "review_required") return "CẦN RÀ SOÁT";
        return "LỖI";
    }

    function setRolloutBadge(status) {
        const el = byId("dashboard-rollout-badge");
        if (!el) return;
        el.className = `text-[10px] font-bold px-2 py-0.5 rounded ${rolloutBadgeClass(status)}`;
        el.textContent = rolloutBadgeLabel(status);
    }

    function renderPassFailMetric(elementId, value) {
        const el = byId(elementId);
        if (!el) return;
        if (value === true) {
            el.textContent = "ĐẠT";
            el.className = "text-xl font-black text-emerald-700 mt-1";
            return;
        }
        if (value === false) {
            el.textContent = "KHÔNG ĐẠT";
            el.className = "text-xl font-black text-red-600 mt-1";
            return;
        }
        el.textContent = "--";
        el.className = "text-xl font-black text-slate-600 mt-1";
    }

    function renderPilotDeltaMetric(elementId, value) {
        const el = byId(elementId);
        if (!el) return;
        const num = Number(value);
        if (!Number.isFinite(num)) {
            el.textContent = "--";
            el.className = "text-xl font-black text-slate-600 mt-1";
            return;
        }
        const sign = num > 0 ? "+" : "";
        el.textContent = `${sign}${num.toFixed(3)}`;
        if (num > 0) {
            el.className = "text-xl font-black text-emerald-700 mt-1";
        } else if (num < 0) {
            el.className = "text-xl font-black text-red-600 mt-1";
        } else {
            el.className = "text-xl font-black text-slate-600 mt-1";
        }
    }

    function updateRolloutWidget(payload) {
        const status = String(payload?.rollout_status || "error").toLowerCase();
        setRolloutBadge(status);

        const summary = payload?.summary || {};
        renderPassFailMetric("dashboard-rollout-hard-gates", summary?.hard_gates_pass);
        renderPassFailMetric("dashboard-rollout-soft-gates", summary?.soft_gates_pass);
        renderPassFailMetric("dashboard-rollout-audit-quality", summary?.audit_quality_pass);
        renderPassFailMetric("dashboard-rollout-vat-quality", summary?.vat_quality_pass);

        const pilot = payload?.artifacts?.pilot || {};
        renderPilotDeltaMetric(
            "dashboard-rollout-audit-delta",
            pilot?.audit_value?.f1_delta_model_minus_heuristic,
        );
        renderPilotDeltaMetric(
            "dashboard-rollout-vat-delta",
            pilot?.vat_refund?.f1_delta_model_minus_heuristic,
        );

        const decision = payload?.artifacts?.go_no_go || {};
        const decisionEl = byId("dashboard-rollout-decision");
        if (decisionEl) {
            const decisionText = String(decision?.decision_status || "unavailable").replaceAll("_", " ").toUpperCase();
            decisionEl.textContent = `${decisionText} • pha_d=${decision?.go_live_phase_d ? "CÓ" : "KHÔNG"}`;
        }

        const messageEl = byId("dashboard-rollout-message");
        if (messageEl) {
            messageEl.textContent = decision?.message || "Chưa có thông điệp go/no-go mới nhất.";
        }

        const actions = Array.isArray(payload?.recommended_actions)
            ? payload.recommended_actions.filter((item) => typeof item === "string" && item.trim())
            : [];
        const actionsEl = byId("dashboard-rollout-actions");
        if (actionsEl) {
            if (!actions.length) {
                actionsEl.innerHTML = '<li class="text-slate-400 italic">Không có hành động được đề xuất.</li>';
            } else {
                actionsEl.innerHTML = actions
                    .slice(0, 4)
                    .map((item) => `<li class="rounded-lg border border-outline-variant/20 bg-white px-3 py-2 leading-relaxed">${escapeHtml(item)}</li>`)
                    .join("");
            }
        }

        const availability = payload?.availability || {};
        const availableCount = [
            availability?.audit_quality,
            availability?.vat_quality,
            availability?.pilot_report,
            availability?.go_no_go_report,
        ].filter(Boolean).length;

        const artifactsEl = byId("dashboard-rollout-artifacts");
        if (artifactsEl) {
            artifactsEl.textContent = `Tệp minh chứng: ${availableCount}/4`;
        }

        const subtextEl = byId("dashboard-rollout-subtext");
        if (subtextEl) {
            const readyText = payload?.phase_d_candidate ? "Ứng viên Pha D" : "Ưu tiên tích hợp";
            subtextEl.textContent = `Trạng thái triển khai ${status.toUpperCase()} • ${readyText}`;
        }

        const updatedEl = byId("dashboard-rollout-updated");
        if (updatedEl) {
            const updated = decision?.updated_at || payload?.generated_at;
            updatedEl.textContent = `Cập nhật: ${formatTimestamp(updated)}`;
        }
    }

    function setRolloutWidgetError(message) {
        setRolloutBadge("error");
        renderPassFailMetric("dashboard-rollout-hard-gates", null);
        renderPassFailMetric("dashboard-rollout-soft-gates", null);
        renderPassFailMetric("dashboard-rollout-audit-quality", null);
        renderPassFailMetric("dashboard-rollout-vat-quality", null);
        renderPilotDeltaMetric("dashboard-rollout-audit-delta", null);
        renderPilotDeltaMetric("dashboard-rollout-vat-delta", null);

        const decisionEl = byId("dashboard-rollout-decision");
        if (decisionEl) decisionEl.textContent = "LỖI";

        const messageEl = byId("dashboard-rollout-message");
        if (messageEl) messageEl.textContent = message || "Không thể tải trạng thái triển khai.";

        const actionsEl = byId("dashboard-rollout-actions");
        if (actionsEl) {
            actionsEl.innerHTML = '<li class="text-red-600 italic">Kiểm tra endpoint /monitoring/specialized_rollout_status.</li>';
        }

        const artifactsEl = byId("dashboard-rollout-artifacts");
        if (artifactsEl) artifactsEl.textContent = "Tệp minh chứng: --";

        const updatedEl = byId("dashboard-rollout-updated");
        if (updatedEl) {
            updatedEl.textContent = `Lỗi lúc: ${new Date().toLocaleTimeString("vi-VN", { hour12: false })}`;
        }
    }

    async function fetchAndRenderRolloutWidget() {
        try {
            const response = await secureFetch(SPECIALIZED_ROLLOUT_ENDPOINT);
            if (!response.ok) {
                throw new Error(`specialized_rollout_status HTTP ${response.status}`);
            }
            const payload = await response.json();
            updateRolloutWidget(payload);
        } catch (error) {
            console.error("Dashboard specialized rollout error:", error);
            setRolloutWidgetError("Không thể kết nối endpoint specialized_rollout_status.");
        }
    }

    function kpiBadgeClass(status) {
        if (status === "ready") return "text-emerald-700 bg-emerald-50";
        if (status === "blocked") return "text-amber-700 bg-amber-50";
        if (status === "schema") return "text-slate-600 bg-slate-100";
        return "text-red-700 bg-red-50";
    }

    function setKpiBadge(status, label) {
        const el = byId("dashboard-kpi-ready-badge");
        if (!el) return;
        el.className = `text-[10px] font-bold px-2 py-0.5 rounded ${kpiBadgeClass(status)}`;
        el.textContent = label;
    }

    function kpiFreshnessClass(status) {
        if (status === "fresh") return "text-emerald-700 bg-emerald-50";
        if (status === "aging") return "text-amber-700 bg-amber-50";
        if (status === "stale") return "text-red-700 bg-red-50";
        return "text-slate-600 bg-slate-100";
    }

    function setKpiFreshnessBadge(status, label) {
        const el = byId("dashboard-kpi-freshness-badge");
        if (!el) return;
        el.className = `text-[10px] font-bold px-2 py-0.5 rounded ${kpiFreshnessClass(status)}`;
        el.textContent = label;
    }

    function normalizeAlertLevel(rawLevel) {
        const normalized = String(rawLevel || "").toLowerCase().trim();
        if (normalized === "critical") return "critical";
        if (normalized === "high") return "high";
        if (normalized === "medium") return "medium";
        if (normalized === "low") return "low";
        return "unknown";
    }

    function kpiAlertLevelBadgeClass(level) {
        if (level === "critical") return "text-red-700 bg-red-100";
        if (level === "high") return "text-orange-700 bg-orange-100";
        if (level === "medium") return "text-amber-700 bg-amber-100";
        if (level === "low") return "text-emerald-700 bg-emerald-100";
        return "text-slate-600 bg-slate-100";
    }

    function kpiAlertLevelCardClass(level) {
        if (level === "critical") return "border-red-300 ring-1 ring-red-200 shadow-red-100/90";
        if (level === "high") return "border-orange-300 ring-1 ring-orange-100 shadow-orange-100/80";
        if (level === "medium") return "border-amber-300 ring-1 ring-amber-100 shadow-amber-100/80";
        if (level === "low") return "border-emerald-200 ring-1 ring-emerald-100 shadow-emerald-100/80";
        return "border-outline-variant/20";
    }

    function kpiAlertLevelPanelClass(level) {
        if (level === "critical") return "border-red-200 bg-red-50/60";
        if (level === "high") return "border-orange-200 bg-orange-50/60";
        if (level === "medium") return "border-amber-200 bg-amber-50/60";
        if (level === "low") return "border-emerald-200 bg-emerald-50/50";
        return "border-outline-variant/20 bg-surface-container-low";
    }

    function setKpiAlertLevel(rawLevel) {
        const level = normalizeAlertLevel(rawLevel);

        const badge = byId("dashboard-kpi-alert-level-badge");
        if (badge) {
            badge.className = `text-[10px] font-bold px-2 py-0.5 rounded transition-colors duration-300 ${kpiAlertLevelBadgeClass(level)}`;
            badge.textContent = `CẢNH BÁO: ${level.toUpperCase()}`;
        }

        const card = byId("dashboard-kpi-governance-card");
        if (card) {
            card.className = `col-span-12 bg-surface-container-lowest rounded-xl shadow-sm overflow-hidden border transition-all duration-300 ${kpiAlertLevelCardClass(level)}`;
        }

        const panel = byId("dashboard-kpi-alert-codes-panel");
        if (panel) {
            panel.className = `rounded-lg border p-4 transition-all duration-300 ${kpiAlertLevelPanelClass(level)}`;
        }
    }

    function kpiAlertCodeClass(active, severity) {
        if (!active) return "border-outline-variant/20 bg-white text-slate-500";
        if (severity === "critical") return "border-red-200 bg-red-50 text-red-700";
        if (severity === "high") return "border-orange-200 bg-orange-50 text-orange-700";
        if (severity === "medium") return "border-amber-200 bg-amber-50 text-amber-700";
        return "border-blue-200 bg-blue-50 text-blue-700";
    }

    function kpiAlertCodeChipClass(active, severity) {
        if (!active) return "bg-slate-100 text-slate-500";
        if (severity === "critical") return "bg-red-100 text-red-700";
        if (severity === "high") return "bg-orange-100 text-orange-700";
        if (severity === "medium") return "bg-amber-100 text-amber-700";
        return "bg-blue-100 text-blue-700";
    }

    function renderKpiAlertCodes(alertPayload) {
        const holder = byId("dashboard-kpi-alert-codes");
        const summaryEl = byId("dashboard-kpi-alert-codes-summary");
        if (!holder) return;

        const alerts = Array.isArray(alertPayload?.alerts) ? alertPayload.alerts : [];
        const alertByCode = new Map();
        alerts.forEach((item) => {
            const code = String(item?.code || "").toLowerCase().trim();
            if (!code || alertByCode.has(code)) return;
            alertByCode.set(code, item);
        });

        let activeCount = 0;
        holder.innerHTML = KPI_WATCH_ALERT_CODES.map((item) => {
            const activeAlert = alertByCode.get(item.code);
            const isActive = Boolean(activeAlert);
            if (isActive) activeCount += 1;

            const severity = String(activeAlert?.severity || "low").toLowerCase();
            const statusText = isActive ? `ĐANG KÍCH HOẠT ${severity.toUpperCase()}` : "ỔN";
            const detailText = isActive
                ? String(activeAlert?.message || item.description)
                : item.normalMessage;

            return `
                <div class="rounded-lg border px-3 py-2 transition-all duration-300 ${kpiAlertCodeClass(isActive, severity)}">
                    <div class="flex items-center justify-between gap-2">
                        <p class="text-[10px] font-black tracking-wide">${escapeHtml(item.code)}</p>
                        <span class="text-[9px] font-bold uppercase px-1.5 py-0.5 rounded ${kpiAlertCodeChipClass(isActive, severity)}">${escapeHtml(statusText)}</span>
                    </div>
                    <p class="text-[11px] mt-1 leading-relaxed">${escapeHtml(detailText)}</p>
                </div>
            `;
        }).join("");

        if (summaryEl) {
            summaryEl.textContent = `Đang kích hoạt ${activeCount}/${KPI_WATCH_ALERT_CODES.length}`;
        }
    }

    function setKpiAlertCodesUnavailable(message) {
        const holder = byId("dashboard-kpi-alert-codes");
        const summaryEl = byId("dashboard-kpi-alert-codes-summary");
        if (summaryEl) {
            summaryEl.textContent = "Không khả dụng";
        }
        if (!holder) return;
        holder.innerHTML = `
            <div class="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-[11px] text-red-700 md:col-span-3">
                ${escapeHtml(message || "Không thể tải danh sách mã cảnh báo quản trị.")}
            </div>
        `;
    }

    function inferSnapshotFreshness(alertPayload) {
        const snapshotAgeHours = Number(alertPayload?.snapshot_freshness?.snapshot_age_hours);
        const staleThreshold = Number(alertPayload?.snapshot_freshness?.stale_snapshot_hours);

        if (!Number.isFinite(snapshotAgeHours) || !Number.isFinite(staleThreshold) || staleThreshold <= 0) {
            return {
                status: "unknown",
                label: "KHÔNG RÕ",
                ageText: "--",
            };
        }

        if (snapshotAgeHours <= staleThreshold) {
            return {
                status: "fresh",
                label: "MỚI",
                ageText: `${snapshotAgeHours.toFixed(1)}h`,
            };
        }

        if (snapshotAgeHours <= staleThreshold * 1.5) {
            return {
                status: "aging",
                label: "ĐANG CŨ",
                ageText: `${snapshotAgeHours.toFixed(1)}h`,
            };
        }

        return {
            status: "stale",
            label: "CŨ",
            ageText: `${snapshotAgeHours.toFixed(1)}h`,
        };
    }

    function metricStatusChip(status) {
        const normalized = String(status || "").toLowerCase();
        if (normalized === "pass") return "bg-emerald-50 text-emerald-700";
        if (normalized === "fail") return "bg-red-50 text-red-700";
        if (normalized === "insufficient_data") return "bg-amber-50 text-amber-700";
        if (normalized === "disabled") return "bg-slate-100 text-slate-600";
        return "bg-slate-100 text-slate-600";
    }

    function metricStatusLabel(status) {
        const normalized = String(status || "").toLowerCase();
        if (normalized === "pass") return "đạt";
        if (normalized === "fail") return "không đạt";
        if (normalized === "insufficient_data") return "thiếu dữ liệu";
        if (normalized === "disabled") return "tắt";
        return normalized || "không rõ";
    }

    function formatMetricValue(metricName, value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) return "--";
        if (String(metricName || "").includes("rate") || String(metricName || "").includes("precision")) {
            return `${(numeric * 100).toFixed(1)}%`;
        }
        return numeric.toFixed(3);
    }

    function renderKpiTrackStatus(splitPayload, alertPayload) {
        const holder = byId("dashboard-kpi-track-status");
        if (!holder) return;

        const trackStatus = splitPayload?.track_status;
        if (!trackStatus || typeof trackStatus !== "object") {
            holder.innerHTML = '<p class="text-slate-400 italic">Không có dữ liệu track status.</p>';
            return;
        }

        const trackPassRateRows = Array.isArray(alertPayload?.track_pass_rates)
            ? alertPayload.track_pass_rates
            : [];
        const trackPassRateMap = new Map(
            trackPassRateRows.map((row) => [String(row?.track_name || ""), row])
        );
        const minTrackPassRate = Number(alertPayload?.pass_rate_summary?.min_track_pass_rate);

        const preferredOrder = ["audit_value", "vat_refund", "intervention"];
        const sortedTracks = Object.keys(trackStatus).sort((left, right) => {
            const leftIndex = preferredOrder.indexOf(left);
            const rightIndex = preferredOrder.indexOf(right);
            const leftRank = leftIndex === -1 ? 999 : leftIndex;
            const rightRank = rightIndex === -1 ? 999 : rightIndex;
            if (leftRank !== rightRank) return leftRank - rightRank;
            return left.localeCompare(right);
        });

        if (!sortedTracks.length) {
            holder.innerHTML = '<p class="text-slate-400 italic">Không có track KPI nào được cấu hình.</p>';
            return;
        }

        holder.innerHTML = sortedTracks
            .map((trackName) => {
                const trackPayload = trackStatus[trackName] || {};
                const rules = Array.isArray(trackPayload.rules) ? trackPayload.rules : [];
                const enabledRules = rules.filter((rule) => Boolean(rule?.enabled));
                const passCount = enabledRules.filter((rule) => String(rule?.status || "") === "pass").length;
                const ready = Boolean(trackPayload.ready_for_split);
                const title = trackName.replaceAll("_", " ").toUpperCase();
                const passRateRow = trackPassRateMap.get(trackName);
                const passRate = Number(passRateRow?.pass_rate);
                const passRateText = Number.isFinite(passRate) ? `${(passRate * 100).toFixed(1)}%` : "--";
                const lowPassRate = Number.isFinite(passRate) && Number.isFinite(minTrackPassRate) && passRate < minTrackPassRate;
                const passRateClass = lowPassRate ? "text-red-600" : "text-slate-500";

                return `
                    <div class="rounded-lg border border-outline-variant/20 bg-white px-3 py-2">
                        <div class="flex items-center justify-between gap-2">
                            <p class="font-bold text-slate-700 tracking-wide">${escapeHtml(title)}</p>
                            <span class="text-[10px] font-bold px-2 py-0.5 rounded ${ready ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}">${ready ? "SẴN SÀNG" : "ĐANG KHÓA"}</span>
                        </div>
                        <p class="text-[11px] text-slate-500 mt-1">Đạt ${passCount}/${enabledRules.length} quy tắc • Đang chặn ${Number(trackPayload.blocking_rule_count || 0)}</p>
                        <p class="text-[11px] mt-1 ${passRateClass}">Tỷ lệ đạt theo track (14 ngày): ${escapeHtml(passRateText)}</p>
                    </div>
                `;
            })
            .join("");
    }

    function renderLatestKpiMetrics(snapshotPayload) {
        const holder = byId("dashboard-kpi-latest-metrics");
        if (!holder) return;

        if (!snapshotPayload || snapshotPayload.available === false) {
            holder.innerHTML = '<p class="text-slate-400 italic">Chưa có dữ liệu snapshot. Hãy bật persist hoặc capture thủ công.</p>';
            return;
        }

        const latestByMetric = Array.isArray(snapshotPayload.latest_by_metric)
            ? snapshotPayload.latest_by_metric
            : [];

        if (!latestByMetric.length) {
            holder.innerHTML = '<p class="text-slate-400 italic">Không có metric snapshot trong 30 ngày gần nhất.</p>';
            return;
        }

        holder.innerHTML = latestByMetric
            .slice(0, 6)
            .map((item) => {
                const track = String(item.track_name || "unknown").replaceAll("_", " ");
                const metric = String(item.metric_name || "metric").replaceAll("_", " ");
                const status = String(item.status || "no_metric").toLowerCase();
                const valueText = formatMetricValue(item.metric_name, item.metric_value);
                const sampleSize = Number(item.sample_size || 0);
                const chipClass = metricStatusChip(status);
                const statusLabelText = metricStatusLabel(status);
                return `
                    <div class="rounded-lg border border-outline-variant/20 bg-white px-3 py-2">
                        <div class="flex items-center justify-between gap-2">
                            <p class="font-semibold text-slate-700">${escapeHtml(track)} • ${escapeHtml(metric)}</p>
                            <span class="text-[10px] font-bold px-2 py-0.5 rounded ${chipClass}">${escapeHtml(statusLabelText)}</span>
                        </div>
                        <p class="text-[11px] text-slate-500 mt-1">Giá trị ${escapeHtml(valueText)} • Mẫu ${sampleSize}</p>
                    </div>
                `;
            })
            .join("");
    }

    function updateKpiWidget(splitPayload, snapshotPayload, alertsPayload, persistSnapshot) {
        const schemaReady = Boolean(splitPayload?.schema_ready);
        const ready = Boolean(splitPayload?.ready);
        const alertLevel = normalizeAlertLevel(alertsPayload?.alert_level);
        setKpiAlertLevel(alertLevel);
        renderKpiAlertCodes(alertsPayload || {});

        if (!schemaReady) {
            setKpiBadge("schema", "LƯỢC ĐỒ");
        } else if (ready) {
            setKpiBadge("ready", "SẴN SÀNG");
        } else {
            setKpiBadge("blocked", "ĐANG KHÓA");
        }

        const freshness = inferSnapshotFreshness(alertsPayload);
        setKpiFreshnessBadge(freshness.status, freshness.label);

        const readinessScoreEl = byId("dashboard-kpi-readiness-score");
        if (readinessScoreEl) {
            const score = Number(splitPayload?.readiness_score);
            readinessScoreEl.textContent = Number.isFinite(score) ? `${score.toFixed(1)}%` : "--";
        }

        const criticalPassEl = byId("dashboard-kpi-critical-pass");
        if (criticalPassEl) {
            const criticalTracks = Array.isArray(splitPayload?.critical_tracks) ? splitPayload.critical_tracks : [];
            const trackStatus = splitPayload?.track_status && typeof splitPayload.track_status === "object"
                ? splitPayload.track_status
                : {};
            const presentCritical = criticalTracks.filter((track) => Object.prototype.hasOwnProperty.call(trackStatus, track));
            const passCount = presentCritical.filter((track) => Boolean(trackStatus[track]?.ready_for_split)).length;
            criticalPassEl.textContent = `${passCount}/${presentCritical.length || criticalTracks.length || 0}`;
        }

        const passRateEl = byId("dashboard-kpi-pass-rate");
        if (passRateEl) {
            const passRate = Number(alertsPayload?.pass_rate_summary?.overall_pass_rate);
            const fallbackPassRate = Number(snapshotPayload?.pass_rate);
            const resolvedPassRate = Number.isFinite(passRate) ? passRate : fallbackPassRate;
            passRateEl.textContent = Number.isFinite(resolvedPassRate) ? `${(resolvedPassRate * 100).toFixed(1)}%` : "--";
        }

        const readinessDriftEl = byId("dashboard-kpi-readiness-drift");
        if (readinessDriftEl) {
            const readinessDrift = Number(alertsPayload?.readiness_summary?.drift_pp);
            readinessDriftEl.textContent = toSignedPoints(readinessDrift);
            if (!Number.isFinite(readinessDrift)) {
                readinessDriftEl.className = "text-2xl font-black text-slate-500 mt-1";
            } else if (readinessDrift <= -8) {
                readinessDriftEl.className = "text-2xl font-black text-red-600 mt-1";
            } else if (readinessDrift < 0) {
                readinessDriftEl.className = "text-2xl font-black text-amber-600 mt-1";
            } else {
                readinessDriftEl.className = "text-2xl font-black text-emerald-700 mt-1";
            }
        }

        const passDriftEl = byId("dashboard-kpi-pass-drift");
        if (passDriftEl) {
            const passDrift = Number(alertsPayload?.pass_rate_summary?.drift_pp);
            passDriftEl.textContent = toSignedPercent(passDrift);
            if (!Number.isFinite(passDrift)) {
                passDriftEl.className = "text-2xl font-black text-slate-500 mt-1";
            } else if (passDrift <= -0.08) {
                passDriftEl.className = "text-2xl font-black text-red-600 mt-1";
            } else if (passDrift < 0) {
                passDriftEl.className = "text-2xl font-black text-amber-600 mt-1";
            } else {
                passDriftEl.className = "text-2xl font-black text-emerald-700 mt-1";
            }
        }

        const latestSnapshotEl = byId("dashboard-kpi-latest-snapshot");
        if (latestSnapshotEl) {
            const latest = alertsPayload?.snapshot_freshness?.latest_generated_at
                || snapshotPayload?.latest_generated_at
                || splitPayload?.generated_at;
            latestSnapshotEl.textContent = formatTimestamp(latest);
        }

        const snapshotAgeEl = byId("dashboard-kpi-snapshot-age");
        if (snapshotAgeEl) {
            const ageHours = Number(alertsPayload?.snapshot_freshness?.snapshot_age_hours);
            const staleHours = Number(alertsPayload?.snapshot_freshness?.stale_snapshot_hours);
            const ageText = Number.isFinite(ageHours) ? `${ageHours.toFixed(1)}h` : "--";
            const thresholdText = Number.isFinite(staleHours) ? `${staleHours}h` : "--";
            snapshotAgeEl.textContent = `Tuổi snapshot: ${ageText} • SLA ${thresholdText}`;
        }

        const trackPassRates = Array.isArray(alertsPayload?.track_pass_rates)
            ? alertsPayload.track_pass_rates
            : [];
        const minTrackPassRate = Number(alertsPayload?.pass_rate_summary?.min_track_pass_rate);
        const lowPassTrackCount = trackPassRates.filter((track) => {
            const rate = Number(track?.pass_rate);
            return Number.isFinite(rate) && Number.isFinite(minTrackPassRate) && rate < minTrackPassRate;
        }).length;

        const totalSnapshotsEl = byId("dashboard-kpi-total-snapshots");
        if (totalSnapshotsEl) {
            const totalSnapshots = Number(snapshotPayload?.total_snapshots);
            const snapshotsText = Number.isFinite(totalSnapshots)
                ? `Tổng snapshot 30 ngày: ${totalSnapshots}`
                : "Tổng snapshot 30 ngày: --";
            totalSnapshotsEl.textContent = `${snapshotsText} • Track thấp ngưỡng: ${lowPassTrackCount}`;
        }

        const topAlert = Array.isArray(alertsPayload?.alerts) && alertsPayload.alerts.length > 0
            ? alertsPayload.alerts[0]
            : null;

        const subtextEl = byId("dashboard-kpi-subtext");
        if (subtextEl) {
            if (splitPayload?.reason) {
                subtextEl.textContent = splitPayload.reason;
            } else {
                const enabledRules = Number(splitPayload?.totals?.enabled_rules || 0);
                const passedRules = Number(splitPayload?.totals?.passed_rules || 0);
                const captured = Number(splitPayload?.snapshots_captured || 0);
                const capturedText = persistSnapshot
                    ? ` • đã lưu ${captured} dòng snapshot`
                    : "";
                const alertText = topAlert?.message
                    ? ` • Cảnh báo vận hành: ${topAlert.message}`
                    : "";
                subtextEl.textContent = `Quy tắc đạt ${passedRules}/${enabledRules}${capturedText}${alertText}`;
            }
        }

        const updatedEl = byId("dashboard-kpi-updated");
        if (updatedEl) {
            updatedEl.textContent = `Cập nhật: ${new Date().toLocaleTimeString("vi-VN", { hour12: false })}`;
        }

        renderKpiTrackStatus(splitPayload || {}, alertsPayload || {});
        renderLatestKpiMetrics(snapshotPayload || {});
    }

    function setKpiWidgetError(message) {
        setKpiBadge("error", "LỖI");
        setKpiFreshnessBadge("unknown", "KHÔNG RÕ");
        setKpiAlertLevel("critical");
        setKpiAlertCodesUnavailable("Không thể tải alert codes từ split_trigger_alerts.");

        const subtextEl = byId("dashboard-kpi-subtext");
        if (subtextEl) {
            subtextEl.textContent = message || "Không thể tải KPI split-trigger từ backend.";
        }

        const readinessScoreEl = byId("dashboard-kpi-readiness-score");
        if (readinessScoreEl) readinessScoreEl.textContent = "--";

        const criticalPassEl = byId("dashboard-kpi-critical-pass");
        if (criticalPassEl) criticalPassEl.textContent = "--/--";

        const passRateEl = byId("dashboard-kpi-pass-rate");
        if (passRateEl) passRateEl.textContent = "--";

        const readinessDriftEl = byId("dashboard-kpi-readiness-drift");
        if (readinessDriftEl) {
            readinessDriftEl.className = "text-2xl font-black text-slate-500 mt-1";
            readinessDriftEl.textContent = "--";
        }

        const passDriftEl = byId("dashboard-kpi-pass-drift");
        if (passDriftEl) {
            passDriftEl.className = "text-2xl font-black text-slate-500 mt-1";
            passDriftEl.textContent = "--";
        }

        const latestSnapshotEl = byId("dashboard-kpi-latest-snapshot");
        if (latestSnapshotEl) latestSnapshotEl.textContent = "--";

        const totalSnapshotsEl = byId("dashboard-kpi-total-snapshots");
        if (totalSnapshotsEl) totalSnapshotsEl.textContent = "Tổng snapshot 30 ngày: --";

        const snapshotAgeEl = byId("dashboard-kpi-snapshot-age");
        if (snapshotAgeEl) snapshotAgeEl.textContent = "Tuổi snapshot: -- • SLA --";

        const trackHolder = byId("dashboard-kpi-track-status");
        if (trackHolder) {
            trackHolder.innerHTML = '<p class="text-red-600 italic">Không thể tải track status.</p>';
        }

        const metricHolder = byId("dashboard-kpi-latest-metrics");
        if (metricHolder) {
            metricHolder.innerHTML = '<p class="text-red-600 italic">Không thể tải KPI snapshots.</p>';
        }

        const updatedEl = byId("dashboard-kpi-updated");
        if (updatedEl) {
            updatedEl.textContent = `Lỗi lúc: ${new Date().toLocaleTimeString("vi-VN", { hour12: false })}`;
        }
    }

    function getPersistToggleState() {
        const toggle = byId("dashboard-kpi-persist-toggle");
        return Boolean(toggle?.checked);
    }

    async function fetchAndRenderKpiWidget() {
        const persistSnapshot = getPersistToggleState();
        const splitUrl = `${KPI_SPLIT_ENDPOINT}?persist_snapshot=${persistSnapshot ? "true" : "false"}&snapshot_source=dashboard_poll`;

        try {
            const [splitResponse, snapshotsResponse, alertsResponse] = await Promise.all([
                secureFetch(splitUrl),
                secureFetch(KPI_SNAPSHOTS_ENDPOINT),
                secureFetch(KPI_ALERTS_ENDPOINT),
            ]);

            if (!splitResponse.ok) {
                throw new Error(`split_trigger_status HTTP ${splitResponse.status}`);
            }

            const splitPayload = await splitResponse.json();
            let snapshotPayload = null;
            let alertsPayload = null;

            if (snapshotsResponse.ok) {
                snapshotPayload = await snapshotsResponse.json();
            } else {
                console.warn("Dashboard KPI snapshots warning:", snapshotsResponse.status);
            }

            if (alertsResponse.ok) {
                alertsPayload = await alertsResponse.json();
            } else {
                console.warn("Dashboard KPI alerts warning:", alertsResponse.status);
            }

            updateKpiWidget(splitPayload, snapshotPayload, alertsPayload, persistSnapshot);
        } catch (error) {
            console.error("Dashboard KPI governance error:", error);
            setKpiWidgetError("Không thể kết nối endpoint split_trigger_status, kpi_snapshots hoặc split_trigger_alerts.");
        }
    }

    function initializeKpiControls() {
        const toggle = byId("dashboard-kpi-persist-toggle");
        const refreshBtn = byId("dashboard-kpi-refresh");
        if (!toggle || !refreshBtn) return;

        try {
            toggle.checked = window.localStorage.getItem(KPI_PERSIST_STORAGE_KEY) === "1";
        } catch {
            toggle.checked = false;
        }

        toggle.addEventListener("change", () => {
            try {
                window.localStorage.setItem(KPI_PERSIST_STORAGE_KEY, toggle.checked ? "1" : "0");
            } catch {
                // Ignore local storage failures.
            }
            fetchAndRenderKpiWidget();
        });

        refreshBtn.addEventListener("click", () => {
            fetchAndRenderKpiWidget();
        });
    }

    function initializeRolloutControls() {
        const refreshBtn = byId("dashboard-rollout-refresh");
        if (!refreshBtn) return;
        refreshBtn.addEventListener("click", () => {
            fetchAndRenderRolloutWidget();
        });
    }

    function refreshDashboardWidgets() {
        if (byId("dashboard-delinquency-health-status")) {
            fetchAndRenderHealth();
        }
        if (byId("dashboard-kpi-readiness-score")) {
            fetchAndRenderKpiWidget();
        }
        if (byId("dashboard-rollout-badge")) {
            fetchAndRenderRolloutWidget();
        }
    }

    function startRealtimePolling() {
        if (pollTimer) {
            clearInterval(pollTimer);
        }
        pollTimer = setInterval(() => {
            if (document.hidden) return;
            refreshDashboardWidgets();
        }, POLL_INTERVAL_MS);
    }

    document.addEventListener("DOMContentLoaded", () => {
        const hasHealthWidget = Boolean(byId("dashboard-delinquency-health-status"));
        const hasKpiWidget = Boolean(byId("dashboard-kpi-readiness-score"));
        const hasRolloutWidget = Boolean(byId("dashboard-rollout-badge"));
        if (!hasHealthWidget && !hasKpiWidget && !hasRolloutWidget) return;

        if (hasKpiWidget) {
            initializeKpiControls();
        }
        if (hasRolloutWidget) {
            initializeRolloutControls();
        }

        refreshDashboardWidgets();
        startRealtimePolling();

        document.addEventListener("visibilitychange", () => {
            if (!document.hidden) {
                refreshDashboardWidgets();
            }
        });
    });
})();
