/**
 * delinquency.js
 * ==============
 * Delinquency list page controller.
 * - Renders record cards from /api/delinquency
 * - Uses URL state for stable pagination/filter navigation
 * - Routes detail action to dedicated detail page
 */

const PAGE_SIZE = 25;
const DELINQ_SPLIT_TRIGGER_ENDPOINT = `${API_BASE}/monitoring/split_trigger_status`;
const ALLOWED_FRESHNESS = new Set(["fresh", "aging", "stale", "unknown"]);
const ALLOWED_INTERVENTION_FILTER = new Set([
    "priority_hot",
    "escalated_enforcement",
    "field_audit",
    "structured_outreach",
    "auto_reminder",
    "monitor",
]);
const ALLOWED_SORT = new Set([
    "risk_desc",
    "intervention_priority_desc",
    "intervention_priority_asc",
    "expected_risk_reduction_desc",
    "expected_collection_uplift_desc",
]);
const SORT_LABELS = {
    risk_desc: "Rủi ro",
    intervention_priority_desc: "Ưu tiên ↓",
    intervention_priority_asc: "Ưu tiên ↑",
    expected_risk_reduction_desc: "Giảm rủi ro",
    expected_collection_uplift_desc: "Tăng thu hồi",
};

const state = {
    currentPage: 1,
    totalPages: 1,
    total: 0,
    searchKeyword: "",
    freshnessFilter: "",
    interventionFilter: "",
    sortBy: "risk_desc",
    splitTriggerGate: null,
};

const filterModalState = {
    isOpen: false,
    focusTarget: "basic",
    previousFocusedElement: null,
    draft: {
        searchKeyword: "",
        freshnessFilter: "",
        interventionFilter: "",
        sortBy: "risk_desc",
    },
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


function formatSplitGateTimestamp(raw) {
    if (!raw) return "--";
    const dt = new Date(raw);
    if (Number.isNaN(dt.getTime())) return "--";
    return dt.toLocaleString("vi-VN", { hour12: false });
}


function normalizeSplitTriggerStatus(rawStatus) {
    const payload = rawStatus && typeof rawStatus === "object" ? rawStatus : {};
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
            reason = "Schema KPI chưa sẵn sàng cho cơ chế quản trị split-trigger.";
        } else if (ready) {
            reason = "Split-trigger đã sẵn sàng. Có thể vận hành đầy đủ các luồng hành động.";
        } else {
            reason = "Split-trigger chưa đạt ngưỡng KPI, hệ thống giữ chế độ theo dõi.";
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


function setElementDisabled(el, disabled) {
    if (!el) return;
    el.disabled = Boolean(disabled);
    if (disabled) {
        el.classList.add("opacity-50", "pointer-events-none");
    } else {
        el.classList.remove("opacity-50", "pointer-events-none");
    }
}


function applyDelinquencySplitTriggerGate(rawStatus) {
    const gate = normalizeSplitTriggerStatus(rawStatus);
    state.splitTriggerGate = gate;

    const badge = document.getElementById("delinq-split-gate-badge");
    const summary = document.getElementById("delinq-split-gate-summary");
    const updated = document.getElementById("delinq-split-gate-updated");

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
        badge.className = `text-[10px] px-2 py-0.5 rounded font-black uppercase tracking-wider ${badgeClass}`;
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

    const lockInterventionActions = !gate.ready;
    const lockReason = gate.reason || "Split-trigger chưa sẵn sàng.";

    if (lockInterventionActions && (state.interventionFilter || state.sortBy !== "risk_desc")) {
        state.interventionFilter = "";
        state.sortBy = "risk_desc";
    }

    const batchButton = document.getElementById("delinq-batch-predict-btn");
    const openSortButton = document.getElementById("delinq-open-sort-btn");

    setElementDisabled(batchButton, lockInterventionActions);

    if (batchButton) {
        batchButton.title = lockInterventionActions ? lockReason : "Chạy dự báo hàng loạt trễ hạn nộp thuế";
    }
    if (openSortButton) {
        openSortButton.title = lockInterventionActions
            ? `${lockReason} (chỉ giữ sắp xếp rủi ro mặc định)`
            : "Mở cấu hình sắp xếp";
    }

    document.querySelectorAll("#delinq-intervention-legend [data-legend-intervention]").forEach((button) => {
        setElementDisabled(button, lockInterventionActions);
        button.title = lockInterventionActions ? lockReason : "Lọc nhanh theo can thiệp";
    });

    syncInterventionLegendState();
    syncModalGovernanceState();
    renderFilterToolbarState();
    syncDelinquencyWorkspaceLink();
}


async function fetchDelinquencySplitTriggerGate() {
    try {
        const response = await secureFetch(DELINQ_SPLIT_TRIGGER_ENDPOINT);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        applyDelinquencySplitTriggerGate(payload);
    } catch (error) {
        console.warn("Split-trigger gate fetch warning:", error);
        applyDelinquencySplitTriggerGate({
            schema_ready: false,
            ready: false,
            reason: "Không tải được trạng thái split-trigger. Dự báo hàng loạt tạm khóa để đảm bảo quản trị.",
            readiness_score: 0,
            totals: { enabled_rules: 0, passed_rules: 0 },
        });
    }
}


function normalizeFreshnessValue(value) {
    const normalized = (value || "").toString().trim().toLowerCase();
    return ALLOWED_FRESHNESS.has(normalized) ? normalized : "";
}


function normalizeSearchKeywordValue(value) {
    return (value || "").toString().trim().slice(0, 80);
}


function normalizeInterventionFilterValue(value) {
    const normalized = (value || "").toString().trim().toLowerCase();
    return ALLOWED_INTERVENTION_FILTER.has(normalized) ? normalized : "";
}


function normalizeSortValue(value) {
    const normalized = (value || "").toString().trim().toLowerCase();
    return ALLOWED_SORT.has(normalized) ? normalized : "risk_desc";
}


function getSortLabel(sortBy) {
    const key = normalizeSortValue(sortBy);
    return SORT_LABELS[key] || SORT_LABELS.risk_desc;
}


function getFreshnessFilterLabel(freshness) {
    const key = normalizeFreshnessValue(freshness);
    if (key === "fresh") return "Mới (0-7 ngày)";
    if (key === "aging") return "Đang cũ (8-30 ngày)";
    if (key === "stale") return "Cũ (>30 ngày)";
    if (key === "unknown") return "Không rõ";
    return "Tất cả";
}


function getInterventionFilterLabel(intervention) {
    const key = normalizeInterventionFilterValue(intervention);
    if (key === "priority_hot") return "Ưu tiên nóng (P≥70)";
    return getInterventionActionLabel(key);
}


function getActiveFilterItems() {
    const chips = [];

    if (state.searchKeyword) {
        chips.push({ key: "search", label: `Từ khóa: ${state.searchKeyword}` });
    }
    if (state.freshnessFilter) {
        chips.push({ key: "freshness", label: `Độ mới: ${getFreshnessFilterLabel(state.freshnessFilter)}` });
    }
    if (state.interventionFilter) {
        chips.push({ key: "intervention", label: `Can thiệp: ${getInterventionFilterLabel(state.interventionFilter)}` });
    }
    if (state.sortBy !== "risk_desc") {
        chips.push({ key: "sort", label: `Sắp xếp: ${getSortLabel(state.sortBy)}` });
    }

    return chips;
}


function renderFilterToolbarState() {
    const activeItems = getActiveFilterItems();
    const countElement = document.getElementById("delinq-active-filter-count");
    if (countElement) {
        countElement.textContent = String(activeItems.length);
    }

    const sortLabel = document.getElementById("delinq-sort-label");
    if (sortLabel) {
        sortLabel.textContent = getSortLabel(state.sortBy);
    }

    const summaryContainer = document.getElementById("delinq-active-filter-summary");
    const chipsContainer = document.getElementById("delinq-active-filter-chips");
    if (!summaryContainer || !chipsContainer) {
        return;
    }

    if (!activeItems.length) {
        chipsContainer.innerHTML = "";
        summaryContainer.classList.add("hidden");
        summaryContainer.classList.remove("flex");
        return;
    }

    chipsContainer.innerHTML = activeItems.map((item) => `
        <button type="button" data-remove-filter="${item.key}" class="delinq-active-filter-chip inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2 py-1 text-[10px] font-bold text-slate-600 hover:text-primary-container hover:border-primary-container/40 transition-colors">
            <span>${escapeHtml(item.label)}</span>
            <span class="material-symbols-outlined text-[12px]">close</span>
        </button>`).join("");

    summaryContainer.classList.remove("hidden");
    summaryContainer.classList.add("flex");
}


function syncModalDraftFromState() {
    filterModalState.draft.searchKeyword = state.searchKeyword;
    filterModalState.draft.freshnessFilter = state.freshnessFilter;
    filterModalState.draft.interventionFilter = state.interventionFilter;
    filterModalState.draft.sortBy = state.sortBy;
}


function syncModalInterventionButtons() {
    const selected = normalizeInterventionFilterValue(filterModalState.draft.interventionFilter);
    document.querySelectorAll("#delinq-modal-intervention-group [data-modal-intervention]").forEach((button) => {
        const value = normalizeInterventionFilterValue(button.getAttribute("data-modal-intervention") || "");
        button.setAttribute("aria-pressed", selected === value ? "true" : "false");
    });
}


function syncModalSortRadios() {
    const selected = normalizeSortValue(filterModalState.draft.sortBy);
    document.querySelectorAll('input[name="delinq-modal-sort"]').forEach((radio) => {
        radio.checked = normalizeSortValue(radio.value) === selected;
    });
}


function syncModalGovernanceState() {
    const gate = state.splitTriggerGate;
    const lockInterventionActions = Boolean(gate && !gate.ready);
    const lockReason = gate?.reason || "Split-trigger chưa sẵn sàng.";

    if (lockInterventionActions) {
        filterModalState.draft.interventionFilter = "";
        filterModalState.draft.sortBy = "risk_desc";
    }

    const governanceHint = document.getElementById("delinq-modal-governance-hint");
    if (governanceHint) {
        if (lockInterventionActions) {
            governanceHint.textContent = lockReason;
            governanceHint.classList.remove("hidden");
        } else {
            governanceHint.classList.add("hidden");
            governanceHint.textContent = "";
        }
    }

    document.querySelectorAll("#delinq-modal-intervention-group [data-modal-intervention]").forEach((button) => {
        const value = normalizeInterventionFilterValue(button.getAttribute("data-modal-intervention") || "");
        const lockButton = lockInterventionActions && value !== "";
        setElementDisabled(button, lockButton);
        button.title = lockButton ? lockReason : "Chọn mức can thiệp";
    });

    document.querySelectorAll('input[name="delinq-modal-sort"]').forEach((radio) => {
        const lockRadio = lockInterventionActions && normalizeSortValue(radio.value) !== "risk_desc";
        radio.disabled = lockRadio;
        radio.title = lockRadio ? lockReason : "Chọn kiểu sắp xếp";
    });

    syncModalInterventionButtons();
    syncModalSortRadios();
}


function renderFilterModalDraft() {
    const modalSearch = document.getElementById("delinq-modal-search-keyword");
    if (modalSearch) {
        modalSearch.value = filterModalState.draft.searchKeyword;
    }

    const modalFreshness = document.getElementById("delinq-modal-freshness");
    if (modalFreshness) {
        modalFreshness.value = filterModalState.draft.freshnessFilter;
    }

    syncModalInterventionButtons();
    syncModalSortRadios();
    syncModalGovernanceState();
}


function getFilterModalFocusableElements() {
    const modal = document.getElementById("delinq-filter-modal");
    if (!modal) return [];

    return Array.from(
        modal.querySelectorAll(
            'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        )
    ).filter((element) => element.offsetParent !== null);
}


function focusFilterModalTarget(target = "basic") {
    if (target === "sort") {
        const sortSection = document.getElementById("delinq-modal-sort-section");
        if (sortSection) {
            sortSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
        const radio = document.querySelector('input[name="delinq-modal-sort"]:not([disabled])');
        if (radio) {
            radio.focus();
            return;
        }
    }

    const searchInput = document.getElementById("delinq-modal-search-keyword");
    if (searchInput) {
        searchInput.focus();
    }
}


function trapFilterModalTab(event) {
    const focusables = getFilterModalFocusableElements();
    if (!focusables.length) {
        event.preventDefault();
        return;
    }

    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement;

    if (event.shiftKey && active === first) {
        event.preventDefault();
        last.focus();
    } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
    }
}


function openFilterModal(target = "basic") {
    const modal = document.getElementById("delinq-filter-modal");
    if (!modal) return;

    filterModalState.isOpen = true;
    filterModalState.focusTarget = target;
    filterModalState.previousFocusedElement = document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;

    syncModalDraftFromState();
    renderFilterModalDraft();

    modal.classList.remove("hidden");
    modal.setAttribute("aria-hidden", "false");
    document.body.classList.add("overflow-hidden");

    window.requestAnimationFrame(() => {
        focusFilterModalTarget(target);
    });
}


function closeFilterModal({ restoreFocus = true } = {}) {
    const modal = document.getElementById("delinq-filter-modal");
    if (!modal) return;

    filterModalState.isOpen = false;
    modal.classList.add("hidden");
    modal.setAttribute("aria-hidden", "true");
    document.body.classList.remove("overflow-hidden");

    if (restoreFocus && filterModalState.previousFocusedElement) {
        filterModalState.previousFocusedElement.focus();
    }
}


function applyFilterModalDraft() {
    state.searchKeyword = normalizeSearchKeywordValue(filterModalState.draft.searchKeyword);
    state.freshnessFilter = normalizeFreshnessValue(filterModalState.draft.freshnessFilter);
    state.interventionFilter = normalizeInterventionFilterValue(filterModalState.draft.interventionFilter);
    state.sortBy = normalizeSortValue(filterModalState.draft.sortBy);

    if (state.splitTriggerGate && !state.splitTriggerGate.ready) {
        state.interventionFilter = "";
        state.sortBy = "risk_desc";
    }

    const mainSearch = document.getElementById("delinq-search-keyword");
    if (mainSearch) {
        mainSearch.value = state.searchKeyword;
    }

    state.currentPage = 1;
    closeFilterModal();
    syncInterventionLegendState();
    renderFilterToolbarState();
    loadDelinquencyData(1);
}


function resetFilterModalDraft() {
    filterModalState.draft.searchKeyword = "";
    filterModalState.draft.freshnessFilter = "";
    filterModalState.draft.interventionFilter = "";
    filterModalState.draft.sortBy = "risk_desc";
    renderFilterModalDraft();
}


function applyQuickFilterUpdate() {
    if (state.splitTriggerGate && !state.splitTriggerGate.ready) {
        state.interventionFilter = "";
        state.sortBy = "risk_desc";
    }

    const searchInput = document.getElementById("delinq-search-keyword");
    if (searchInput) {
        searchInput.value = state.searchKeyword;
    }

    state.currentPage = 1;
    syncInterventionLegendState();
    syncModalDraftFromState();
    renderFilterModalDraft();
    renderFilterToolbarState();
    loadDelinquencyData(1);
}


function removeActiveFilterByKey(filterKey) {
    if (filterKey === "search") {
        state.searchKeyword = "";
    } else if (filterKey === "freshness") {
        state.freshnessFilter = "";
    } else if (filterKey === "intervention") {
        state.interventionFilter = "";
    } else if (filterKey === "sort") {
        state.sortBy = "risk_desc";
    }
    applyQuickFilterUpdate();
}


function attachFilterToolbarHandlers() {
    const openFilterButton = document.getElementById("delinq-open-filter-btn");
    if (openFilterButton) {
        openFilterButton.addEventListener("click", () => openFilterModal("basic"));
    }

    const openSortButton = document.getElementById("delinq-open-sort-btn");
    if (openSortButton) {
        openSortButton.addEventListener("click", () => openFilterModal("sort"));
    }

    const activeFilterSummary = document.getElementById("delinq-active-filter-summary");
    if (activeFilterSummary) {
        activeFilterSummary.addEventListener("click", (event) => {
            const clearButton = event.target.closest("#delinq-clear-filters-btn");
            if (clearButton) {
                state.searchKeyword = "";
                state.freshnessFilter = "";
                state.interventionFilter = "";
                state.sortBy = "risk_desc";
                applyQuickFilterUpdate();
                return;
            }

            const removeChip = event.target.closest("button[data-remove-filter]");
            if (!removeChip) {
                return;
            }

            removeActiveFilterByKey(removeChip.getAttribute("data-remove-filter") || "");
        });
    }
}


function attachFilterModalHandlers() {
    const modal = document.getElementById("delinq-filter-modal");
    if (!modal) return;

    modal.addEventListener("click", (event) => {
        const closeTarget = event.target.closest("[data-filter-modal-close]");
        if (closeTarget) {
            closeFilterModal();
        }
    });

    modal.addEventListener("keydown", (event) => {
        if (event.key === "Tab") {
            trapFilterModalTab(event);
        }
        if (event.key === "Enter") {
            const tagName = String(event.target?.tagName || "").toLowerCase();
            if (tagName === "input" || tagName === "select") {
                event.preventDefault();
                applyFilterModalDraft();
            }
        }
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && filterModalState.isOpen) {
            event.preventDefault();
            closeFilterModal();
        }
    });

    const modalSearch = document.getElementById("delinq-modal-search-keyword");
    if (modalSearch) {
        modalSearch.addEventListener("input", (event) => {
            filterModalState.draft.searchKeyword = normalizeSearchKeywordValue(event.target.value);
        });
    }

    const modalFreshness = document.getElementById("delinq-modal-freshness");
    if (modalFreshness) {
        modalFreshness.addEventListener("change", (event) => {
            filterModalState.draft.freshnessFilter = normalizeFreshnessValue(event.target.value);
        });
    }

    const interventionGroup = document.getElementById("delinq-modal-intervention-group");
    if (interventionGroup) {
        interventionGroup.addEventListener("click", (event) => {
            const button = event.target.closest("button[data-modal-intervention]");
            if (!button || button.disabled) return;

            filterModalState.draft.interventionFilter = normalizeInterventionFilterValue(button.getAttribute("data-modal-intervention") || "");
            syncModalInterventionButtons();
        });
    }

    modal.querySelectorAll('input[name="delinq-modal-sort"]').forEach((radio) => {
        radio.addEventListener("change", (event) => {
            if (!event.target.checked) return;
            filterModalState.draft.sortBy = normalizeSortValue(event.target.value);
            syncModalSortRadios();
        });
    });

    const resetButton = document.getElementById("delinq-modal-reset-btn");
    if (resetButton) {
        resetButton.addEventListener("click", resetFilterModalDraft);
    }

    const applyButton = document.getElementById("delinq-modal-apply-btn");
    if (applyButton) {
        applyButton.addEventListener("click", applyFilterModalDraft);
    }
}


function hasClientSideInterventionControls() {
    return Boolean(state.interventionFilter) || state.sortBy !== "risk_desc" || Boolean(state.searchKeyword);
}


function parseInitialStateFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const rawPage = Number(params.get("page") || "1");
    const parsedPage = Number.isInteger(rawPage) && rawPage > 0 ? rawPage : 1;

    return {
        page: parsedPage,
        searchKeyword: normalizeSearchKeywordValue(params.get("q")),
        freshness: normalizeFreshnessValue(params.get("freshness")),
        interventionFilter: normalizeInterventionFilterValue(params.get("intervention_filter")),
        sortBy: normalizeSortValue(params.get("sort_by")),
    };
}


function buildInterventionWorkspaceUrl() {
    const params = new URLSearchParams();
    params.set("source", "delinquency");
    params.set("focus", "actions");
    params.set("window_days", "90");
    params.set("top_k", "50");

    if (state.currentPage > 1) {
        params.set("page", String(state.currentPage));
    }
    if (state.searchKeyword) {
        params.set("q", state.searchKeyword);
    }
    if (state.freshnessFilter) {
        params.set("freshness", state.freshnessFilter);
    }
    if (state.interventionFilter) {
        params.set("intervention_filter", state.interventionFilter);
    }
    if (state.sortBy && state.sortBy !== "risk_desc") {
        params.set("sort_by", state.sortBy);
    }

    return `intervention.html?${params.toString()}`;
}


function syncDelinquencyWorkspaceLink() {
    const link = document.getElementById("delinq-open-intervention-workspace");
    if (!link) return;
    link.href = buildInterventionWorkspaceUrl();
}


function syncUrlState() {
    const params = new URLSearchParams(window.location.search);

    if (state.currentPage > 1) {
        params.set("page", String(state.currentPage));
    } else {
        params.delete("page");
    }

    if (state.searchKeyword) {
        params.set("q", state.searchKeyword);
    } else {
        params.delete("q");
    }

    if (state.freshnessFilter) {
        params.set("freshness", state.freshnessFilter);
    } else {
        params.delete("freshness");
    }

    if (state.interventionFilter) {
        params.set("intervention_filter", state.interventionFilter);
    } else {
        params.delete("intervention_filter");
    }

    if (state.sortBy && state.sortBy !== "risk_desc") {
        params.set("sort_by", state.sortBy);
    } else {
        params.delete("sort_by");
    }

    const query = params.toString();
    const nextUrl = `${window.location.pathname}${query ? `?${query}` : ""}`;
    window.history.replaceState({}, "", nextUrl);
    syncDelinquencyWorkspaceLink();
}


function syncInterventionLegendState() {
    const legendButtons = document.querySelectorAll("#delinq-intervention-legend [data-legend-intervention]");
    if (!legendButtons.length) return;

    legendButtons.forEach((button) => {
        const value = normalizeInterventionFilterValue(button.getAttribute("data-legend-intervention") || "");
        const selected = (state.interventionFilter || "") === value;
        button.setAttribute("aria-pressed", selected ? "true" : "false");
    });
}


function applyInterventionLegendSelection(rawValue) {
    if (state.splitTriggerGate && !state.splitTriggerGate.ready) {
        setBatchStatus(state.splitTriggerGate.reason || "Split-trigger chưa sẵn sàng.", "warning");
        return;
    }

    state.interventionFilter = normalizeInterventionFilterValue(rawValue);
    state.currentPage = 1;

    if (state.interventionFilter === "priority_hot" && state.sortBy === "risk_desc") {
        state.sortBy = "intervention_priority_desc";
    }

    syncInterventionLegendState();
    syncModalDraftFromState();
    renderFilterModalDraft();
    renderFilterToolbarState();
    loadDelinquencyData(1);
}


function attachInterventionLegendHandler() {
    const legend = document.getElementById("delinq-intervention-legend");
    if (!legend) return;

    legend.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-legend-intervention]");
        if (!button) return;
        applyInterventionLegendSelection(button.getAttribute("data-legend-intervention") || "");
    });

    const legendToggleButton = document.getElementById("delinq-legend-toggle-btn");
    const legendScroll = document.getElementById("delinq-intervention-legend-scroll");
    if (legendToggleButton && legendScroll) {
        legendToggleButton.addEventListener("click", () => {
            const collapsed = legendScroll.classList.toggle("hidden");
            legendToggleButton.textContent = collapsed ? "Hiện chú giải" : "Ẩn chú giải";
        });
    }
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
        return { label: "Mới", classes: "bg-emerald-50 text-emerald-700 border border-emerald-100" };
    }
    if (key === "aging") {
        return { label: "Đang cũ", classes: "bg-amber-50 text-amber-700 border border-amber-100" };
    }
    if (key === "stale") {
        return { label: "Cũ", classes: "bg-rose-50 text-rose-700 border border-rose-100" };
    }
    return { label: "Không rõ", classes: "bg-slate-100 text-slate-600 border border-slate-200" };
}


function getSourceBadge(scoreSource) {
    const key = (scoreSource || "unknown").toString().toLowerCase();
    if (key === "ml_model") {
        return { label: "ML", classes: "bg-blue-50 text-blue-700 border border-blue-100" };
    }
    if (key === "statistical_baseline") {
        return { label: "Đường cơ sở", classes: "bg-violet-50 text-violet-700 border border-violet-100" };
    }
    if (key === "no_data") {
        return { label: "Không có dữ liệu", classes: "bg-slate-100 text-slate-600 border border-slate-200" };
    }
    return { label: "Dự phòng", classes: "bg-orange-50 text-orange-700 border border-orange-100" };
}


function normalizeEarlyWarning(rawWarning) {
    const payload = rawWarning && typeof rawWarning === "object" ? rawWarning : {};
    const queueRaw = String(payload.queue || "monitor").toLowerCase();
    const levelRaw = String(payload.level || "low").toLowerCase();

    const queue = queueRaw === "priority_review" || queueRaw === "watchlist" ? queueRaw : "monitor";
    const level = ["critical", "high", "medium", "low"].includes(levelRaw) ? levelRaw : "low";
    const tags = Array.isArray(payload.tags) ? payload.tags.map((tag) => String(tag || "").trim()).filter(Boolean).slice(0, 3) : [];
    const reason = String(payload.reason || "").trim();
    const hasWarning = Boolean(payload.has_warning) || level !== "low" || queue !== "monitor";

    return {
        hasWarning,
        queue,
        level,
        tags,
        reason,
    };
}


function earlyWarningVisual(earlyWarning) {
    if (!earlyWarning.hasWarning) {
        return {
            queueLabel: "Theo dõi",
            queueClass: "bg-slate-100 text-slate-600 border border-slate-200",
            levelClass: "text-slate-500",
        };
    }

    if (earlyWarning.queue === "priority_review") {
        return {
            queueLabel: "Ưu tiên rà soát",
            queueClass: "bg-rose-50 text-rose-700 border border-rose-100",
            levelClass: "text-rose-700",
        };
    }

    if (earlyWarning.queue === "watchlist") {
        return {
            queueLabel: "Danh sách theo dõi",
            queueClass: "bg-amber-50 text-amber-700 border border-amber-100",
            levelClass: "text-amber-700",
        };
    }

    return {
        queueLabel: "Theo dõi",
        queueClass: "bg-blue-50 text-blue-700 border border-blue-100",
        levelClass: "text-blue-700",
    };
}


function renderEarlyWarningChip(item) {
    const earlyWarning = normalizeEarlyWarning(item?.early_warning);
    const visual = earlyWarningVisual(earlyWarning);
    const reason = earlyWarning.reason || "Chưa ghi nhận tín hiệu cảnh báo sớm đáng kể.";

    const tagsHtml = earlyWarning.tags.length
        ? `<div class="flex flex-wrap gap-1 mt-1">${earlyWarning.tags
            .map((tag) => `<span class="text-[9px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-500 font-semibold">${escapeHtml(tag)}</span>`)
            .join("")}</div>`
        : "";

    return `
        <div class="mt-2 rounded-lg border border-slate-100 bg-slate-50 px-2.5 py-2">
            <div class="flex items-center justify-between gap-2">
                <span class="text-[9px] font-black uppercase tracking-wider px-2 py-0.5 rounded ${visual.queueClass}">${visual.queueLabel}</span>
                <span class="text-[10px] font-bold uppercase ${visual.levelClass}">${escapeHtml(earlyWarning.level)}</span>
            </div>
            <p class="text-[10px] leading-relaxed text-slate-600 mt-1">${escapeHtml(reason)}</p>
            ${tagsHtml}
        </div>`;
}


function normalizeInterventionUplift(rawIntervention, item) {
    const payload = rawIntervention && typeof rawIntervention === "object" ? rawIntervention : {};
    const probability = Math.max(0, Math.min(1, toSafeNumber(item?.probability, 0)));

    const actionRaw = String(payload.recommended_action || "").trim().toLowerCase();
    const supportedActions = [
        "monitor",
        "auto_reminder",
        "structured_outreach",
        "field_audit",
        "escalated_enforcement",
    ];
    const fallbackAction = probability >= 0.8
        ? "escalated_enforcement"
        : probability >= 0.6
            ? "field_audit"
            : probability >= 0.4
                ? "structured_outreach"
                : probability >= 0.2
                    ? "auto_reminder"
                    : "monitor";
    const action = supportedActions.includes(actionRaw) ? actionRaw : fallbackAction;

    const priorityScoreRaw = toSafeNumber(payload.priority_score, Math.round(probability * 100));
    const priorityScore = Math.max(0, Math.min(100, Math.round(priorityScoreRaw)));

    const expectedRiskReduction = Math.max(0, toSafeNumber(payload.expected_risk_reduction_pp, probability * 14));
    const expectedCollectionUplift = Math.max(0, toSafeNumber(payload.expected_collection_uplift, 0));

    return {
        action,
        priorityScore,
        expectedRiskReduction,
        expectedCollectionUplift,
    };
}


function getInterventionActionLabel(action) {
    if (action === "escalated_enforcement") return "Cưỡng chế tăng cường";
    if (action === "field_audit") return "Kiểm tra thực địa";
    if (action === "structured_outreach") return "Tiếp cận có cấu trúc";
    if (action === "auto_reminder") return "Nhắc hạn tự động";
    return "Theo dõi";
}


function getInterventionActionClass(action) {
    if (action === "escalated_enforcement") return "bg-rose-50 text-rose-700 border border-rose-100";
    if (action === "field_audit") return "bg-orange-50 text-orange-700 border border-orange-100";
    if (action === "structured_outreach") return "bg-amber-50 text-amber-700 border border-amber-100";
    if (action === "auto_reminder") return "bg-blue-50 text-blue-700 border border-blue-100";
    return "bg-slate-100 text-slate-600 border border-slate-200";
}


function renderInterventionQuickChip(item) {
    const intervention = normalizeInterventionUplift(item?.intervention_uplift, item);
    return `
        <div class="mt-2 rounded-lg border border-slate-100 bg-slate-50 px-2.5 py-2 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-sm">
            <div class="flex items-center justify-between gap-2">
                <span class="text-[9px] font-black uppercase tracking-wider px-2 py-0.5 rounded ${getInterventionActionClass(intervention.action)}">${escapeHtml(getInterventionActionLabel(intervention.action))}</span>
                <span class="text-[10px] font-bold text-slate-600">P${intervention.priorityScore}</span>
            </div>
            <div class="flex flex-wrap items-center gap-1.5 mt-1.5">
                <span class="text-[9px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 font-semibold">-${intervention.expectedRiskReduction.toFixed(1)}pp</span>
                <span class="text-[9px] px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 font-semibold">+${escapeHtml(formatCurrencyCompact(intervention.expectedCollectionUplift))}</span>
            </div>
        </div>`;
}


function applyInterventionControls(items) {
    if (!Array.isArray(items) || !items.length) {
        return [];
    }

    const withSignals = items.map((item) => ({
        ...item,
        __intervention: normalizeInterventionUplift(item?.intervention_uplift, item),
    }));

    let filtered = withSignals;
    const keyword = (state.searchKeyword || "").toLowerCase();
    if (keyword) {
        filtered = filtered.filter((item) => {
            const taxCode = String(item?.tax_code || "").toLowerCase();
            const companyName = String(item?.company_name || item?.name || "").toLowerCase();
            const cluster = String(item?.cluster || "").toLowerCase();
            return taxCode.includes(keyword) || companyName.includes(keyword) || cluster.includes(keyword);
        });
    }

    if (state.interventionFilter) {
        if (state.interventionFilter === "priority_hot") {
            filtered = filtered.filter((item) => Number(item.__intervention?.priorityScore || 0) >= 70);
        } else {
            filtered = filtered.filter((item) => String(item.__intervention?.action || "") === state.interventionFilter);
        }
    }

    const sorted = [...filtered];
    sorted.sort((left, right) => {
        const leftRisk = toSafeNumber(left?.probability ?? left?.risk_probability, 0);
        const rightRisk = toSafeNumber(right?.probability ?? right?.risk_probability, 0);
        const leftIntervention = left.__intervention || {};
        const rightIntervention = right.__intervention || {};

        if (state.sortBy === "intervention_priority_desc") {
            const diff = Number(rightIntervention.priorityScore || 0) - Number(leftIntervention.priorityScore || 0);
            return diff !== 0 ? diff : rightRisk - leftRisk;
        }
        if (state.sortBy === "intervention_priority_asc") {
            const diff = Number(leftIntervention.priorityScore || 0) - Number(rightIntervention.priorityScore || 0);
            return diff !== 0 ? diff : rightRisk - leftRisk;
        }
        if (state.sortBy === "expected_risk_reduction_desc") {
            const diff = Number(rightIntervention.expectedRiskReduction || 0) - Number(leftIntervention.expectedRiskReduction || 0);
            return diff !== 0 ? diff : rightRisk - leftRisk;
        }
        if (state.sortBy === "expected_collection_uplift_desc") {
            const diff = Number(rightIntervention.expectedCollectionUplift || 0) - Number(leftIntervention.expectedCollectionUplift || 0);
            return diff !== 0 ? diff : rightRisk - leftRisk;
        }

        return rightRisk - leftRisk;
    });

    return sorted.map((item) => {
        delete item.__intervention;
        return item;
    });
}


function buildDetailUrl(taxCode) {
    const params = new URLSearchParams();
    params.set("tax_code", taxCode);
    params.set("page", String(state.currentPage));
    if (state.searchKeyword) {
        params.set("q", state.searchKeyword);
    }
    if (state.freshnessFilter) {
        params.set("freshness", state.freshnessFilter);
    }
    if (state.interventionFilter) {
        params.set("intervention_filter", state.interventionFilter);
    }
    if (state.sortBy && state.sortBy !== "risk_desc") {
        params.set("sort_by", state.sortBy);
    }
    return `delinquency-detail.html?${params.toString()}`;
}


function renderPredictionMeta(item) {
    const freshness = getFreshnessBadge(item.freshness);
    const source = getSourceBadge(item.score_source);
    const ageDays = Number.isFinite(Number(item.prediction_age_days)) ? Number(item.prediction_age_days) : null;
    const ageText = ageDays === null ? "Không có" : `${ageDays} ngày`;

    return `
        <div class="flex flex-wrap items-center gap-1.5 mt-2">
            <span class="text-[10px] px-2 py-0.5 rounded-full font-bold ${freshness.classes}">${freshness.label}</span>
            <span class="text-[10px] px-2 py-0.5 rounded-full font-bold ${source.classes}">${source.label}</span>
            <span class="text-[10px] text-slate-500 font-semibold">Tuổi dự báo: ${escapeHtml(ageText)}</span>
            ${item.monotonic_adjusted ? '<span class="text-[10px] px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-semibold">Đã hiệu chỉnh đơn điệu</span>' : ""}
        </div>`;
}


function renderProbabilityBreakdown(item) {
    const p30 = toSafeNumber(item.prob_30d, 0);
    const p60 = toSafeNumber(item.prob_60d, 0);
    const p90 = toSafeNumber(item.prob_90d, 0);

    return `
        <div class="flex gap-1.5 mt-2 flex-wrap">
            <span class="text-[10px] px-2 py-0.5 rounded ${p30 > 0.5 ? "bg-red-50 text-red-600" : "bg-slate-50 text-slate-500"} font-bold">
                30 ngày: ${(p30 * 100).toFixed(0)}%
            </span>
            <span class="text-[10px] px-2 py-0.5 rounded ${p60 > 0.5 ? "bg-orange-50 text-orange-600" : "bg-slate-50 text-slate-500"} font-bold">
                60 ngày: ${(p60 * 100).toFixed(0)}%
            </span>
            <span class="text-[10px] px-2 py-0.5 rounded ${p90 > 0.5 ? "bg-yellow-50 text-yellow-700" : "bg-slate-50 text-slate-500"} font-bold">
                90 ngày: ${(p90 * 100).toFixed(0)}%
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
            <span class="text-[10px] px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 font-semibold">Đúng hạn ${onTime}</span>
            <span class="text-[10px] px-2 py-0.5 rounded bg-orange-50 text-orange-700 font-semibold">Trễ hạn ${late}</span>
            ${unpaid > 0 ? `<span class="text-[10px] px-2 py-0.5 rounded bg-red-50 text-red-700 font-semibold">Chưa nộp ${unpaid}</span>` : ""}
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
        const interventionSignal = normalizeInterventionUplift(item?.intervention_uplift, item);
        const interventionLabel = getInterventionActionLabel(interventionSignal.action);
        const interventionClass = getInterventionActionClass(interventionSignal.action);

        const paymentSummary = renderPaymentSummary(item.payment_history_summary);
        const predictionMeta = renderPredictionMeta(item);
        const probabilityBreakdown = renderProbabilityBreakdown(item);
        const topReasons = renderTopReasons(item.top_reasons);
        const earlyWarning = renderEarlyWarningChip(item);
        const intervention = renderInterventionQuickChip(item);

        const penaltyAmount = toSafeNumber(item.payment_history_summary?.total_penalties, 0);
        const penaltyText = penaltyAmount > 0 ? `${formatCurrencyCompact(penaltyAmount)} phạt` : "---";
        const detailHref = taxCodeRaw === "---" ? "" : buildDetailUrl(taxCodeRaw);

        return `
            <article data-expandable-row class="px-8 py-4 ${index % 2 === 0 ? "bg-white" : "bg-slate-50/40"} hover:bg-slate-50 transition-colors duration-200">
                <div class="grid grid-cols-12 gap-3 items-center">
                    <div class="col-span-6 sm:col-span-4 md:col-span-2">
                        <p class="text-[10px] tracking-wide text-slate-400 font-bold">MST</p>
                        <p class="font-mono text-sm text-slate-600 mt-1">${taxCode}</p>
                    </div>
                    <div class="col-span-12 sm:col-span-8 md:col-span-3">
                        <h4 class="font-bold text-primary-container text-[15px] leading-tight">${companyName}</h4>
                        <p class="mt-1 text-[11px] text-slate-500 font-semibold">${cluster}</p>
                    </div>
                    <div class="col-span-12 md:col-span-3">
                        <p class="text-[10px] tracking-wide text-slate-400 font-bold">Xác suất tổng</p>
                        <div class="flex items-center gap-2 mt-1.5">
                            <div class="w-28 h-2 bg-slate-100 rounded-full overflow-hidden">
                                <div class="h-full ${riskStyle.bg} rounded-full transition-all duration-500" style="width:${widthPct}%"></div>
                            </div>
                            <span class="px-2 py-0.5 ${riskStyle.text} rounded text-[11px] font-bold min-w-[44px] text-center">${widthPct}%</span>
                        </div>
                    </div>
                    <div class="col-span-8 md:col-span-2">
                        <p class="text-[10px] tracking-wide text-slate-400 font-bold">Can thiệp</p>
                        <div class="mt-1.5 flex items-center gap-1.5">
                            <span class="text-[10px] px-2 py-0.5 rounded font-bold ${interventionClass}">${escapeHtml(interventionLabel)}</span>
                            <span class="text-[10px] font-bold text-slate-500">P${interventionSignal.priorityScore}</span>
                        </div>
                    </div>
                    <div class="col-span-4 md:col-span-1">
                        <p class="text-[10px] tracking-wide text-slate-400 font-bold">Phạt</p>
                        <p class="font-bold text-primary-container mt-1 text-sm">${escapeHtml(penaltyText)}</p>
                    </div>
                    <div class="col-span-12 md:col-span-1 flex md:flex-col md:items-end gap-2">
                        ${detailHref
            ? `<a href="${escapeHtml(detailHref)}" class="bg-primary-container text-white px-3 py-1.5 rounded-lg text-xs font-bold hover:opacity-90 transition-all duration-200 inline-flex items-center gap-1">
                                <span class="material-symbols-outlined text-[14px]">visibility</span>
                                Chi tiết
                            </a>`
            : `<span class="text-xs text-slate-400 font-semibold">Không có</span>`}
                        <button type="button" data-expand-toggle aria-expanded="false" class="inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-[10px] font-bold text-slate-500 hover:text-primary-container hover:border-primary-container/40 transition-colors">
                            <span data-expand-label>Mở thêm</span>
                            <span data-expand-icon class="material-symbols-outlined text-[14px]">expand_more</span>
                        </button>
                    </div>
                </div>

                <div data-expand-body class="hidden mt-3 pt-3 border-t border-slate-100">
                    <div class="grid grid-cols-1 xl:grid-cols-3 gap-3">
                        <section class="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
                            <p class="text-[10px] font-bold tracking-wide text-slate-400">Thông tin mô hình</p>
                            <div class="mt-1 flex items-center gap-1.5 flex-wrap">
                                <span class="text-[10px] px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-semibold">Nhóm: ${cluster}</span>
                                ${modelVersion}
                            </div>
                            ${predictionMeta}
                            ${paymentSummary}
                        </section>

                        <section class="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
                            <p class="text-[10px] font-bold tracking-wide text-slate-400">Cảnh báo sớm</p>
                            ${earlyWarning}
                        </section>

                        <section class="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
                            <p class="text-[10px] font-bold tracking-wide text-slate-400">Chi tiết xác suất</p>
                            ${probabilityBreakdown}
                            ${topReasons}
                            ${intervention}
                        </section>
                    </div>
                </div>
            </article>`;
    }).join("");
}


function attachDelinquencyRowExpandHandler() {
    const listContainer = document.getElementById("delinq-list");
    if (!listContainer || listContainer.dataset.expandBound === "true") {
        return;
    }

    listContainer.dataset.expandBound = "true";
    listContainer.addEventListener("click", (event) => {
        const toggleButton = event.target.closest("button[data-expand-toggle]");
        if (!toggleButton) return;

        const row = toggleButton.closest("article[data-expandable-row]");
        const expandBody = row ? row.querySelector("[data-expand-body]") : null;
        if (!expandBody) return;

        const expanded = toggleButton.getAttribute("aria-expanded") === "true";
        toggleButton.setAttribute("aria-expanded", expanded ? "false" : "true");
        expandBody.classList.toggle("hidden", expanded);

        const label = toggleButton.querySelector("[data-expand-label]");
        const icon = toggleButton.querySelector("[data-expand-icon]");
        if (label) {
            label.textContent = expanded ? "Mở thêm" : "Thu gọn";
        }
        if (icon) {
            icon.textContent = expanded ? "expand_more" : "expand_less";
        }
    });
}


function updateKPIs(data, visibleCount = null) {
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
                kpi3El.textContent = modelVersion ? `ML (${modelVersion}, ${staleCount} bản ghi cũ)` : `ML (${staleCount} bản ghi cũ)`;
            } else {
                kpi3El.textContent = modelVersion ? `ML ${modelVersion}` : "Mô hình ML";
            }
        } else if (source === "statistical_baseline") {
            kpi3El.textContent = modelVersion ? `Đường cơ sở ${modelVersion}` : "Đường cơ sở";
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

        const pageCount = Array.isArray(predictions) ? predictions.length : 0;
        const visible = Number.isFinite(Number(visibleCount)) ? Number(visibleCount) : pageCount;
        const hasInterventionControls = hasClientSideInterventionControls();

        if (hasInterventionControls) {
            const keywordSuffix = state.searchKeyword ? ` • "${state.searchKeyword}"` : "";
            paginationInfo.textContent = `Trang ${state.currentPage} • ${visible}/${pageCount} hồ sơ hiển thị • Tổng ${total.toLocaleString("vi-VN")}${keywordSuffix}`;
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

    const requestedPageRaw = Number.isInteger(Number(page)) ? Math.max(1, Number(page)) : 1;
    const enforceClientSideMode = hasClientSideInterventionControls();
    const requestedPage = enforceClientSideMode ? 1 : requestedPageRaw;
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

        state.currentPage = enforceClientSideMode ? 1 : normalizedPage;
        state.total = total;
        state.totalPages = enforceClientSideMode ? 1 : calculatedTotalPages;

        const normalizedData = {
            ...data,
            total,
            page: state.currentPage,
            page_size: PAGE_SIZE,
        };

        const basePredictions = Array.isArray(normalizedData.predictions) ? normalizedData.predictions : [];
        const displayPredictions = applyInterventionControls(basePredictions);
        const paginationTotal = enforceClientSideMode ? displayPredictions.length : total;

        renderDelinquencyRows(listContainer, displayPredictions);
        updateKPIs(normalizedData, displayPredictions.length);
        renderPagination(paginationTotal, state.currentPage, PAGE_SIZE);
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
    const gate = state.splitTriggerGate;
    if (!gate || !gate.ready) {
        setBatchStatus(gate?.reason || "Split-trigger chưa sẵn sàng. Dự báo hàng loạt đang tạm khóa.", "warning");
        return;
    }

    const button = document.getElementById("delinq-batch-predict-btn");
    if (!button) return;

    button.disabled = true;
    button.classList.add("opacity-70", "pointer-events-none");
    setBatchStatus("Đang chạy dự báo hàng loạt trễ hạn nộp thuế...", "info");

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
            `Hoàn tất: đã xử lý=${toSafeNumber(result.processed, 0)}, tạo mới=${toSafeNumber(result.created, 0)}, cập nhật=${toSafeNumber(result.updated, 0)}, lỗi=${failed}`,
            failed > 0 ? "warning" : "success"
        );

        await loadDelinquencyData(state.currentPage);
        await fetchDelinquencySplitTriggerGate();
    } catch (error) {
        console.error("Batch predict error:", error);
        setBatchStatus("Dự báo hàng loạt thất bại. Vui lòng thử lại.", "error");
    } finally {
        button.disabled = false;
        button.classList.remove("opacity-70", "pointer-events-none");
    }
}


document.addEventListener("DOMContentLoaded", () => {
    const initial = parseInitialStateFromUrl();
    state.currentPage = initial.page;
    state.searchKeyword = initial.searchKeyword;
    state.freshnessFilter = initial.freshness;
    state.interventionFilter = initial.interventionFilter;
    state.sortBy = initial.sortBy;

    const searchInput = document.getElementById("delinq-search-keyword");
    let searchDebounce = null;
    if (searchInput) {
        searchInput.value = state.searchKeyword;
        searchInput.addEventListener("input", (event) => {
            if (searchDebounce) {
                window.clearTimeout(searchDebounce);
            }
            searchDebounce = window.setTimeout(() => {
                state.searchKeyword = normalizeSearchKeywordValue(event.target.value);
                state.currentPage = 1;
                syncModalDraftFromState();
                renderFilterModalDraft();
                renderFilterToolbarState();
                loadDelinquencyData(1);
            }, 220);
        });
    }

    const batchButton = document.getElementById("delinq-batch-predict-btn");
    if (batchButton) {
        batchButton.addEventListener("click", runDelinquencyBatchPredict);
    }

    attachFilterToolbarHandlers();
    attachFilterModalHandlers();
    attachInterventionLegendHandler();
    attachDelinquencyRowExpandHandler();

    syncModalDraftFromState();
    renderFilterModalDraft();
    renderFilterToolbarState();
    syncInterventionLegendState();
    syncDelinquencyWorkspaceLink();
    attachPaginationHandler();
    fetchDelinquencySplitTriggerGate();
    loadDelinquencyData(state.currentPage);
});
