(function () {
    const POLL_INTERVAL_MS = 60000;
    const ENDPOINT = `${API_BASE}/delinquency/health/cache?fresh_days=7&stale_days=30`;

    let pollTimer = null;

    function byId(id) {
        return document.getElementById(id);
    }

    function toPercent(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return "N/A";
        return `${(num * 100).toFixed(1)}%`;
    }

    function statusLabel(status) {
        const normalized = (status || "").toString().toLowerCase();
        if (normalized === "healthy") return "HEALTHY";
        if (normalized === "warning") return "WARNING";
        if (normalized === "critical") return "CRITICAL";
        if (normalized === "no_data") return "NO DATA";
        return "ERROR";
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
            subtextEl.textContent = `Coverage ${coverageText}, Stale ${staleText}, Alerts ${alertCount}`;
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
                alertEl.textContent = "Khong co canh bao stale/coverage tai thoi diem hien tai.";
                alertEl.className = "mt-3 text-[11px] text-emerald-700 min-h-[2.75rem]";
            }
        }

        const updatedEl = byId("dashboard-delinquency-health-updated");
        if (updatedEl) {
            updatedEl.textContent = `Cap nhat: ${new Date().toLocaleTimeString("vi-VN")}`;
        }
    }

    function setWidgetError(message) {
        setBadge("error");
        setProgress("error", 0);

        const statusEl = byId("dashboard-delinquency-health-status");
        if (statusEl) statusEl.textContent = "ERROR";

        const subtextEl = byId("dashboard-delinquency-health-subtext");
        if (subtextEl) subtextEl.textContent = "Khong the tai trang thai health tu backend.";

        const alertEl = byId("dashboard-delinquency-health-alert");
        if (alertEl) {
            alertEl.className = "mt-3 text-[11px] text-red-600 min-h-[2.75rem]";
            alertEl.textContent = message || "Vui long kiem tra API /api/delinquency/health/cache.";
        }

        const updatedEl = byId("dashboard-delinquency-health-updated");
        if (updatedEl) updatedEl.textContent = `Loi luc: ${new Date().toLocaleTimeString("vi-VN")}`;
    }

    async function fetchAndRenderHealth() {
        try {
            const response = await secureFetch(ENDPOINT);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const payload = await response.json();
            updateHealthWidget(payload);
        } catch (error) {
            console.error("Dashboard delinquency health error:", error);
            setWidgetError("Khong the ket noi endpoint health.");
        }
    }

    function startRealtimePolling() {
        if (pollTimer) {
            clearInterval(pollTimer);
        }
        pollTimer = setInterval(() => {
            if (document.hidden) return;
            fetchAndRenderHealth();
        }, POLL_INTERVAL_MS);
    }

    document.addEventListener("DOMContentLoaded", () => {
        if (!byId("dashboard-delinquency-health-status")) return;
        fetchAndRenderHealth();
        startRealtimePolling();

        document.addEventListener("visibilitychange", () => {
            if (!document.hidden) {
                fetchAndRenderHealth();
            }
        });
    });
})();
