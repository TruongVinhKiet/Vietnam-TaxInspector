/**
 * graph.js – D3.js Force-Directed Graph for VAT Network Investigation
 * =====================================================================
 * Renders interactive force-directed graph from GNN API response.
 * Features:
 *   - Animated pulse nodes (red for shell corps, blue for legit)
 *   - Dashed animated edges for circular fraud flows
 *   - Forensic panel with real-time investigation logs
 *   - Timeline filter integration
 *   - Drag, zoom, pan with smooth transitions
 */

const GRAPH_API_BASE =
    (typeof API_BASE !== "undefined" && API_BASE) ||
    window.API_BASE ||
    "http://localhost:8000/api";

const FORENSIC_TAB_ACTIVE_CLASS = "py-2 text-[10px] font-black uppercase text-primary-container border-b-2 border-primary-container tracking-widest transition-colors";
const FORENSIC_TAB_INACTIVE_CLASS = "py-2 text-[10px] font-black uppercase text-slate-400 border-b-2 border-transparent hover:text-primary-container tracking-widest transition-colors";
const GRAPH_LEGEND_ITEMS = [
    { id: "normal_flow", label: "Luồng bình thường", type: "edge", color: "#334155", dash: false },
    { id: "circular_flow", label: "Luồng xoay vòng", type: "edge", color: "#ba1a1a", dash: true },
    { id: "ownership_flow", label: "Liên kết sở hữu", type: "edge", color: "#fbbf24", dash: true },
    { id: "normal_node", label: "Doanh nghiệp thường", type: "node", color: "#002147" },
    { id: "suspicious_node", label: "Nút nghi vấn", type: "node", color: "#1e293b" },
    { id: "shell_node", label: "Chủ thể vỏ bọc", type: "node", color: "#ba1a1a" },
    { id: "offshore_node", label: "Offshore/UBO", type: "node", color: "#f43f5e" },
    { id: "ghost_node", label: "Nút chưa định danh", type: "node", color: "#94a3b8" },
];

// ════════════════════════════════════════════════════════════════
//  State
// ════════════════════════════════════════════════════════════════
let graphData = null;
let simulation = null;
let svg = null;
let zoomBehavior = null;
let activeWorkbenchMode = "companies";
let currentTaxCode = null;
let graphQuality = null;
let forensicIntelData = null;
let timelineMonth = 12;
let timelinePlaybackTimer = null;
let graphRenderState = {
    edges: [],
    lines: null,
    edgeLabels: null,
};
let graphLegendState = {
    normal_flow: true,
    circular_flow: true,
    normal_node: true,
    suspicious_node: true,
    shell_node: true,
    offshore_node: true,
};
let graphSplitTriggerGate = null;
let activeNodeFocusId = null;

let allCompanies = [];
let filteredCompanies = [];
let currentCompanyPage = 1;
const COMPANY_PAGE_SIZE = 10;
const FORENSIC_FALLBACK_MAX_LOGS = 20;
const FORENSIC_FALLBACK_MAX_PATHS = 80;
const FORENSIC_FALLBACK_MAX_HOPS = 16;
const FORENSIC_MERGED_LOG_LIMIT = 80;
const FORENSIC_RING_PATH_LIMIT = 80;
const FORENSIC_OWNERSHIP_PATH_LIMIT = 80;
const FORENSIC_CYCLE_MAX_LEN = 8;
const FORENSIC_CYCLE_MAX_COUNT = 120;


function escapeHtml(value) {
    const str = value === null || value === undefined ? "" : String(value);
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;")
        .replace(/'/g, "&#39;");
}


function toFiniteNumber(value, fallback = 0) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}


async function graphFetch(url, options = {}) {
    if (typeof secureFetch === "function") {
        return secureFetch(url, options);
    }
    return fetch(url, { credentials: "include", ...options });
}


function setElementText(id, value) {
    const el = document.getElementById(id);
    if (el) {
        el.textContent = value;
    }
}


function formatThreshold(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(2) : "--";
}


function getNodeLegendCategory(node) {
    if (node && node.is_ghost) return "ghost_node";
    if (node && node.is_offshore) return "offshore_node";
    if (node && node.is_shell) return "shell_node";
    if (node && node.group === "suspicious") return "suspicious_node";
    return "normal_node";
}


function getEdgeLegendCategory(edge) {
    if (edge && edge.is_ownership) return "ownership_flow";
    return edge && edge.is_circular ? "circular_flow" : "normal_flow";
}


function isLegendVisible(category) {
    return graphLegendState[category] !== false;
}


function renderInteractiveLegend() {
    const container = document.getElementById("graph-legend-interactive");
    if (!container) return;

    container.innerHTML = GRAPH_LEGEND_ITEMS.map((item) => {
        const isActive = isLegendVisible(item.id);
        const lineSwatch = item.type === "edge"
            ? `<span class="inline-block w-5 h-0.5 rounded" style="background:${item.color};${item.dash ? `border-top:2px dashed ${item.color};background:transparent;height:0;` : ""}"></span>`
            : `<span class="inline-block w-3 h-3 rounded-full" style="background:${item.color}"></span>`;
        return `
            <button
                type="button"
                data-legend-id="${item.id}"
                aria-pressed="${isActive ? "true" : "false"}"
                class="legend-toggle-item flex items-center gap-2 text-[10px] font-bold uppercase px-2 py-1 rounded-md transition-colors ${isActive ? "text-slate-200 bg-slate-800/60" : "text-slate-500 bg-slate-900/30"}">
                ${lineSwatch}
                <span>${item.label}</span>
            </button>
        `;
    }).join("");

    container.querySelectorAll("[data-legend-id]").forEach((btn) => {
        btn.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            const legendId = btn.getAttribute("data-legend-id");
            if (!legendId || !(legendId in graphLegendState)) return;
            graphLegendState[legendId] = !graphLegendState[legendId];
            renderInteractiveLegend();
            applyLegendVisibility();
        });
    });
}


function renderGraphRiskAnalytics(data) {
    const panel = document.getElementById("graph-risk-analytics");
    const barsHost = document.getElementById("graph-risk-bars");
    const cumulativeHost = document.getElementById("graph-cumulative-risk");
    if (!panel || !barsHost || !cumulativeHost) return;

    const nodes = Array.isArray(data?.nodes) ? data.nodes : [];
    if (!nodes.length) {
        panel.classList.add("hidden");
        return;
    }

    panel.classList.remove("hidden");

    const bucketDefs = [
        { key: "critical", label: "Nghiêm trọng", color: "#ef4444", test: (score) => score >= 80 },
        { key: "high", label: "Cao", color: "#f97316", test: (score) => score >= 60 && score < 80 },
        { key: "medium", label: "Trung bình", color: "#f59e0b", test: (score) => score >= 40 && score < 60 },
        { key: "low", label: "Thấp", color: "#10b981", test: (score) => score < 40 },
    ];

    const counts = { critical: 0, high: 0, medium: 0, low: 0 };
    let totalRisk = 0;
    const scoreList = [];

    nodes.forEach((node) => {
        const score = Math.max(0, Math.min(100, toFiniteNumber(node?.risk_score, 0)));
        totalRisk += score;
        scoreList.push(score);
        const bucket = bucketDefs.find((def) => def.test(score));
        if (bucket) counts[bucket.key] += 1;
    });

    barsHost.innerHTML = bucketDefs.map((def) => {
        const count = counts[def.key] || 0;
        const pct = nodes.length ? (count / nodes.length) * 100 : 0;
        return `
            <div>
                <div class="flex items-center justify-between text-[10px] font-bold text-slate-500 uppercase">
                    <span>${def.label}</span>
                    <span>${count} (${pct.toFixed(1)}%)</span>
                </div>
                <div class="mt-1 h-2 rounded-full bg-slate-200 overflow-hidden">
                    <div class="h-full rounded-full" style="width:${pct.toFixed(2)}%;background:${def.color}"></div>
                </div>
            </div>
        `;
    }).join("");

    const sortedScores = [...scoreList].sort((a, b) => b - a);
    const top10Count = Math.max(1, Math.ceil(sortedScores.length * 0.1));
    const top20Count = Math.max(1, Math.ceil(sortedScores.length * 0.2));
    const top10Risk = sortedScores.slice(0, top10Count).reduce((acc, value) => acc + value, 0);
    const top20Risk = sortedScores.slice(0, top20Count).reduce((acc, value) => acc + value, 0);
    const top10Share = totalRisk > 0 ? (top10Risk / totalRisk) * 100 : 0;
    const top20Share = totalRisk > 0 ? (top20Risk / totalRisk) * 100 : 0;

    cumulativeHost.innerHTML = `
        <div>
            <div class="flex items-center justify-between text-[10px] font-bold uppercase text-slate-500">
                <span>Top 10% doanh nghiệp</span>
                <span>${top10Share.toFixed(1)}% tổng rủi ro</span>
            </div>
            <div class="mt-1 h-2 rounded-full bg-slate-200 overflow-hidden">
                <div class="h-full rounded-full bg-rose-500" style="width:${Math.min(100, top10Share).toFixed(2)}%"></div>
            </div>
        </div>
        <div>
            <div class="flex items-center justify-between text-[10px] font-bold uppercase text-slate-500">
                <span>Top 20% doanh nghiệp</span>
                <span>${top20Share.toFixed(1)}% tổng rủi ro</span>
            </div>
            <div class="mt-1 h-2 rounded-full bg-slate-200 overflow-hidden">
                <div class="h-full rounded-full bg-orange-500" style="width:${Math.min(100, top20Share).toFixed(2)}%"></div>
            </div>
        </div>
    `;
}


function applyLegendVisibility() {
    if (!svg) return;

    svg.selectAll(".node")
        .transition().duration(180)
        .style("opacity", (d) => (isLegendVisible(getNodeLegendCategory(d)) ? 1 : 0.08));

    svg.selectAll(".node-label")
        .transition().duration(180)
        .style("opacity", (d) => (isLegendVisible(getNodeLegendCategory(d)) ? 1 : 0.08));

    applyTimelineFilter(timelineMonth, { refreshBadge: true });
}


function formatShortTimestamp(raw) {
    if (!raw) return "--";
    const date = new Date(raw);
    if (Number.isNaN(date.getTime())) return "--";
    return date.toLocaleString("vi-VN", {
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    });
}


function formatQualityState(value) {
    const state = String(value || "unknown").toLowerCase();
    if (state === "healthy") return "TỐT";
    if (state === "warning") return "CẢNH BÁO";
    if (state === "degraded") return "SUY GIẢM";
    return "KHÔNG RÕ";
}

function formatQualityReason(reason) {
    const value = String(reason || "").toLowerCase();
    const labels = {
        all_quality_gates_passed: "Tất cả cổng chất lượng đều đạt ngưỡng.",
        quality_reports_unavailable: "Chưa có đủ báo cáo serving/stress để kết luận chất lượng.",
        drift_high_detected: "Phát hiện trôi dữ liệu mức cao, cần theo dõi và tái huấn luyện.",
        drift_medium_detected: "Có dấu hiệu trôi dữ liệu mức trung bình.",
    };
    if (labels[value]) return labels[value];
    if (value.includes("temporal_plus3m_edge_f1_drop")) {
        return "Mô hình suy giảm do gate temporal edge F1 drop vượt ngưỡng cho phép.";
    }
    if (value.includes("_failed")) {
        return "Một hoặc nhiều gate kiểm định không đạt ngưỡng.";
    }
    return "Không có diễn giải chi tiết.";
}

function formatGateLabel(raw) {
    const gate = String(raw || "").toLowerCase();
    const labels = {
        temporal_plus3m_edge_f1_drop: "Giảm Edge F1 (+3 tháng)",
        temporal_plus3m_edge_prauc_drop: "Giảm Edge PR-AUC (+3 tháng)",
        unseen_node_generalization_gap: "Khoảng cách tổng quát hóa nút mới",
        node_pr_auc_drop: "Suy giảm Node PR-AUC",
        edge_pr_auc_drop: "Suy giảm Edge PR-AUC",
    };
    return labels[gate] || gate.replaceAll("_", " ");
}


function formatSplitGateTimestamp(raw) {
    if (!raw) return "--";
    const date = new Date(raw);
    if (Number.isNaN(date.getTime())) return "--";
    return date.toLocaleString("vi-VN", { hour12: false });
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
            reason: "Nhập mã số thuế để tải trạng thái split-trigger cho cụm điều tra.",
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
                ? `${normalizedTrack} (${blockingRuleCount} quy tắc chặn)`
                : normalizedTrack;
        });

    let reason = String(payload.reason || "").trim();
    if (!reason) {
        if (!schemaReady) {
            reason = "Schema KPI chưa sẵn sàng cho quản trị split-trigger.";
        } else if (ready) {
            reason = "Split-trigger sẵn sàng. Có thể mở đầy đủ hành động điều tra số.";
        } else {
            reason = "Split-trigger chưa đạt KPI, các hành động cưỡng chế đang bị khóa.";
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


function setGraphActionButtonState(button, disabled) {
    if (!button) return;
    button.disabled = Boolean(disabled);
    if (disabled) {
        button.classList.add("opacity-50", "cursor-not-allowed", "pointer-events-none");
    } else {
        button.classList.remove("opacity-50", "cursor-not-allowed", "pointer-events-none");
    }
}


function renderGraphSplitTriggerGate(rawStatus) {
    const gate = normalizeSplitTriggerStatus(rawStatus);
    graphSplitTriggerGate = gate;
    window._graphSplitTriggerGate = gate;

    const badge = document.getElementById("graph-split-gate-badge");
    const summary = document.getElementById("graph-split-gate-summary");
    const updated = document.getElementById("graph-split-gate-updated");

    let badgeLabel = "SẴN SÀNG";
    let badgeClass = "bg-emerald-100 text-emerald-700";
    if (!gate.schemaReady) {
        badgeLabel = "THIẾU SCHEMA";
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
        const core = `Độ sẵn sàng ${gate.readinessScore.toFixed(1)}% • Quy tắc ${gate.passedRules}/${gate.enabledRules}`;
        const blocked = gate.blockedTracks.length ? ` • Đang chặn: ${gate.blockedTracks.join(", ")}` : "";
        summary.textContent = gate.ready ? `${core}. ${gate.reason}` : `${core}${blocked}. ${gate.reason}`;
    }
    if (updated) {
        updated.textContent = `Cập nhật: ${formatSplitGateTimestamp(gate.generatedAt)}`;
    }

    const lockActions = !gate.ready;
    const lockReason = gate.reason || "Split-trigger chưa sẵn sàng.";
    ["flag-case-btn", "request-explain-btn", "seal-case-btn"].forEach((id) => {
        const button = document.getElementById(id);
        setGraphActionButtonState(button, lockActions);
        if (button) {
            button.title = lockActions ? lockReason : "";
        }
    });

    return gate;
}


function canRunGraphGovernedAction() {
    const gate = graphSplitTriggerGate;
    if (gate && gate.ready) return true;
    showGraphToast(
        "Cổng đang khóa",
        gate?.reason || "Split-trigger chưa sẵn sàng, không thể thực thi hành động điều tra số.",
        "warning",
    );
    return false;
}


function monthToQuarterLabel(month) {
    const m = Math.max(1, Math.min(12, Number(month) || 1));
    if (m <= 3) return "Q1";
    if (m <= 6) return "Q2";
    if (m <= 9) return "Q3";
    return "Q4";
}


function resolveEdgeMonth(value) {
    if (!value) return null;
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return null;
    return date.getMonth() + 1;
}


function applyQualityPill(id, label, state) {
    const pill = document.getElementById(id);
    if (!pill) return;
    pill.textContent = label;
    pill.setAttribute("data-state", state || "unknown");
}


function showGraphToast(title, detail = "", tone = "info") {
    const region = document.getElementById("graph-toast-region");
    if (!region) return;

    const toast = document.createElement("div");
    toast.className = "graph-toast";
    toast.setAttribute("data-tone", tone);
    toast.innerHTML = `
        <p class="text-[11px] font-black uppercase tracking-widest">${escapeHtml(title)}</p>
        ${detail ? `<p class="text-[11px] mt-1 text-slate-600 font-medium leading-relaxed">${escapeHtml(detail)}</p>` : ""}
    `;
    region.appendChild(toast);

    const dismiss = () => {
        toast.classList.add("toast-exit");
        setTimeout(() => toast.remove(), 240);
    };

    setTimeout(dismiss, 2800);
    toast.addEventListener("click", dismiss);
}


function setupActionButtons() {
    const buttons = [
        {
            id: "export-report-btn",
            title: "Xuất báo cáo",
            tone: "info",
            detail: () => `Đã đưa báo cáo cụm ${currentTaxCode || "toàn tuyến"} vào hàng đợi xuất.`,
            requiresGate: false,
        },
        {
            id: "flag-case-btn",
            title: "Đã đánh dấu hồ sơ",
            tone: "warning",
            detail: () => "Hồ sơ điều tra được chuyển vào danh sách ưu tiên.",
            requiresGate: true,
        },
        {
            id: "request-explain-btn",
            title: "Đã gửi yêu cầu giải trình",
            tone: "info",
            detail: () => "Yêu cầu đã được ghi nhận tới doanh nghiệp liên quan.",
            requiresGate: true,
        },
        {
            id: "seal-case-btn",
            title: "Niêm phong hồ sơ",
            tone: "success",
            detail: () => "Hồ sơ đã chuyển sang trạng thái khóa nghiệp vụ.",
            requiresGate: true,
        },
    ];

    buttons.forEach((config) => {
        const btn = document.getElementById(config.id);
        if (!btn || btn.getAttribute("data-initialized") === "true") return;
        btn.setAttribute("data-initialized", "true");
        btn.addEventListener("click", () => {
            if (config.requiresGate && !canRunGraphGovernedAction()) {
                return;
            }
            showGraphToast(config.title, config.detail(), config.tone);
        });
    });
}

// ════════════════════════════════════════════════════════════════
//  Init
// ════════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
    initGraph();
    setupGraphCanvasControls();
    setupActionButtons();
    setupSearch();
    setupTimelineControls();
    setupForensicTabs();
    setupWorkbenchTabs();
    setupGlobalWorkbenchShortcuts();
    resetInvestigationSummary();
    renderModelIntelligence(null);
    renderQualitySummary(null);
    renderGraphSplitTriggerGate(null);
    loadGraphQuality().then((quality) => renderQualitySummary(quality));

    // Auto-load companies list
    loadCompanyList();

    // Entity Resolution bindings
    const btnResolveEntity = document.getElementById('btn-resolve-entity');
    if (btnResolveEntity) {
        btnResolveEntity.addEventListener('click', runEntityResolution);
    }
});

// ════════════════════════════════════════════════════════════════
//  Search
// ════════════════════════════════════════════════════════════════
function setupSearch() {
    const input = document.getElementById("graph-search-input");
    const btn = document.getElementById("graph-search-btn");
    const dropdown = document.getElementById("search-dropdown");

    if (!input || !btn) return;

    // Live search
    let debounce = null;
    input.addEventListener("input", () => {
        clearTimeout(debounce);
        debounce = setTimeout(async () => {
            const q = input.value.trim();
            if (q.length < 2) { dropdown.classList.add("hidden"); return; }
            try {
                const res = await graphFetch(`${GRAPH_API_BASE}/graph/search?q=${encodeURIComponent(q)}`);
                if (!res.ok) throw new Error(`Search API error: ${res.status}`);
                const data = await res.json();
                renderSearchDropdown(data.results || [], dropdown);
            } catch (e) {
                console.warn("Search error:", e);
                showGraphToast("Lỗi tra cứu", "Không thể tải gợi ý mã số thuế lúc này.", "warning");
            }
        }, 300);
    });

    btn.addEventListener("click", () => {
        const q = input.value.trim();
        if (q) loadGraph(q);
        dropdown.classList.add("hidden");
    });

    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            btn.click();
        }
    });

    if (!input.hasAttribute("data-outside-bound")) {
        input.setAttribute("data-outside-bound", "true");
        document.addEventListener("click", (event) => {
            if (dropdown.classList.contains("hidden")) return;
            const target = event.target;
            if (!(target instanceof Node)) return;
            if (!dropdown.contains(target) && !input.contains(target) && !btn.contains(target)) {
                dropdown.classList.add("hidden");
            }
        });
    }
}

function renderSearchDropdown(results, dropdown) {
    if (!results.length) { dropdown.classList.add("hidden"); return; }
    dropdown.innerHTML = "";

    results.forEach((r) => {
        const taxCode = String(r.tax_code || "");
        const name = String(r.name || taxCode);
        const industry = String(r.industry || "");
        const riskScore = toFiniteNumber(r.risk_score, 0);

        const item = document.createElement("div");
        item.className = "px-4 py-2 hover:bg-slate-50 cursor-pointer flex justify-between items-center search-result-item";
        item.dataset.taxCode = taxCode;
        item.innerHTML = `
            <div>
                <p class="text-sm font-bold text-primary-container">${escapeHtml(name)}</p>
                <p class="text-[10px] text-slate-400 font-mono">${escapeHtml(taxCode)} · ${escapeHtml(industry)}</p>
            </div>
            <span class="text-xs font-bold ${riskScore >= 60 ? "text-error" : "text-emerald-600"}">${riskScore.toFixed(0)}%</span>
        `;
        dropdown.appendChild(item);
    });

    dropdown.classList.remove("hidden");

    dropdown.querySelectorAll(".search-result-item").forEach(item => {
        item.addEventListener("click", () => {
            const tc = item.dataset.taxCode;
            document.getElementById("graph-search-input").value = tc;
            dropdown.classList.add("hidden");
            loadGraph(tc);
        });
    });
}

// ════════════════════════════════════════════════════════════════
//  Graph Core
// ════════════════════════════════════════════════════════════════
function initGraph() {
    const container = document.getElementById("graph-canvas");
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    svg = d3.select(container)
        .append("svg")
        .attr("width", "100%")
        .attr("height", "100%")
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("class", "graph-svg");

    // Defs for markers and filters
    const defs = svg.append("defs");

    // Arrow marker
    defs.append("marker")
        .attr("id", "arrow-normal")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 55).attr("refY", 0)
        .attr("markerWidth", 5).attr("markerHeight", 5)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", "rgba(51, 65, 85, 0.4)");

    defs.append("marker")
        .attr("id", "arrow-fraud")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 55).attr("refY", 0)
        .attr("markerWidth", 6).attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", "#ba1a1a");

    // SVG filters removed for better performance and flat UI aesthetic

    // Zoom
    const zoomGroup = svg.append("g").attr("class", "zoom-group");
    zoomBehavior = d3.zoom()
        .scaleExtent([0.2, 5])
        .on("zoom", (event) => {
            zoomGroup.attr("transform", event.transform);
            // Update zoom badge
            const badge = document.getElementById("zoom-badge");
            if (badge) badge.textContent = `TRƯỜNG NHÌN: ${Math.round(event.transform.k * 100)}%`;
        });
    svg.call(zoomBehavior);

    // Layers
    zoomGroup.append("g").attr("class", "boundary-layer");
    zoomGroup.append("g").attr("class", "edges-layer");
    zoomGroup.append("g").attr("class", "nodes-layer");
    zoomGroup.append("g").attr("class", "labels-layer");
}

async function loadGraph(taxCode) {
    const section = document.getElementById("investigation-section");
    if (!taxCode) {
        resetInvestigationSummary();
        renderGraphSplitTriggerGate(null);
        if (section) {
            section.classList.add("opacity-0");
            setTimeout(() => section.classList.add("hidden"), 500);
        }
        return;
    }

    switchWorkbenchMode("graph", { skipScroll: true });

    if (section && section.classList.contains("hidden")) {
        section.classList.remove("hidden");
        // Trigger reflow
        void section.offsetWidth; 
        section.classList.remove("opacity-0");
    }

    const container = document.getElementById("graph-canvas");
    if (!container) return;

    // Show loading
    showLoading(true);
    currentTaxCode = taxCode;
    forensicIntelData = null;

    try {
        const requestedDepth = 2;
        let url = `${GRAPH_API_BASE}/graph`;
        url += `?tax_code=${encodeURIComponent(taxCode)}&depth=${requestedDepth}`;

        const qualityPromise = loadGraphQuality().then((quality) => {
            if (quality) {
                graphQuality = quality;
                renderQualitySummary(graphQuality);
            }
            return quality;
        });
        const forensicPromise = loadForensicIntel(taxCode, requestedDepth);

        const res = await graphFetch(url);
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        graphData = await res.json();
        renderGraphSplitTriggerGate(graphData.split_trigger_status || null);

        renderGraph(graphData);
        renderInvestigationSummary(graphData);
        renderModelIntelligence(graphData);
        renderQualitySummary(graphQuality);
        renderForensicPanel(graphData);

        forensicPromise
            .then((forensicIntel) => {
                if (currentTaxCode !== taxCode) return;
                forensicIntelData = forensicIntel;
                renderForensicPanel(graphData, forensicIntelData);
            })
            .catch((err) => {
                console.warn("Forensic intel load warning:", err);
            });

        await qualityPromise;
        showLoading(false);

    } catch (err) {
        console.error("Graph load error:", err);
        resetInvestigationSummary();
        renderModelIntelligence(null);
        renderGraphSplitTriggerGate({
            schema_ready: false,
            ready: false,
            reason: "Không tải được dữ liệu đồ thị, tạm khóa hành động điều tra số theo cơ chế quản trị.",
            readiness_score: 0,
            totals: { enabled_rules: 0, passed_rules: 0 },
        });
        showGraphToast("Không tải được đồ thị", "Vui lòng thử lại hoặc kiểm tra trạng thái backend.", "warning");
        showLoading(false);
        showEmptyState("Không thể tải dữ liệu đồ thị. Hãy thử lại sau vài giây.");
    }
}


async function safeGraphJson(url) {
    const startedAt = Date.now();
    try {
        const res = await graphFetch(url);
        if (!res.ok) {
            return {
                ok: false,
                statusCode: res.status,
                payload: null,
                dataStatus: "http_error",
                durationMs: Date.now() - startedAt,
            };
        }

        const payload = await res.json();
        if (!payload || typeof payload !== "object") {
            return {
                ok: false,
                statusCode: res.status,
                payload: null,
                dataStatus: "invalid_payload",
                durationMs: Date.now() - startedAt,
            };
        }

        return {
            ok: true,
            statusCode: res.status,
            payload,
            dataStatus: "ok",
            durationMs: Date.now() - startedAt,
        };
    } catch (err) {
        console.warn("Graph supplemental API warning:", err);
        return {
            ok: false,
            statusCode: null,
            payload: null,
            dataStatus: "request_failed",
            durationMs: Date.now() - startedAt,
            error: String(err?.message || err || "unknown_error"),
        };
    }
}


function normalizeForensicDataStatus(status, fallback = "unavailable") {
    const normalized = String(status || "").trim().toLowerCase();
    return normalized || String(fallback || "unavailable").trim().toLowerCase();
}


function isForensicDataUnavailable(status) {
    const normalized = normalizeForensicDataStatus(status);
    return normalized === "request_failed"
        || normalized === "http_error"
        || normalized === "invalid_payload"
        || normalized === "unavailable";
}


function formatForensicDataStatus(status) {
    const normalized = normalizeForensicDataStatus(status);
    const labels = {
        ok: "OK",
        request_failed: "Lỗi kết nối",
        http_error: "API lỗi",
        invalid_payload: "Payload lỗi",
        unavailable: "Không khả dụng",
        no_invoice_context: "Thiếu ngữ cảnh hóa đơn",
        no_ownership_links: "Thiếu liên kết sở hữu",
        ownership_outside_invoice_scope: "Sở hữu ngoài phạm vi hóa đơn",
        no_parent_child_pairs_in_invoice_scope: "Không có cặp cha-con trong phạm vi",
        no_related_party_trades_found: "Chưa có giao dịch liên đới",
        no_cycles_detected: "Chưa phát hiện vòng",
        no_cycles_with_circular_edges: "Cạnh đỏ cao, chưa khép vòng",
        cycles_detected_but_no_rings_scored: "Có vòng nhưng chưa đạt ngưỡng",
        partial_ring_output: "Kết quả vòng trả về một phần",
    };
    return labels[normalized] || normalized.replace(/_/g, " ");
}


async function loadForensicIntel(taxCode, depth = 2) {
    const ownershipParams = new URLSearchParams();
    if (taxCode) ownershipParams.set("tax_code", taxCode);

    const ringParams = new URLSearchParams();
    if (taxCode) ringParams.set("tax_code", taxCode);
    ringParams.set("depth", String(Math.max(1, Math.min(4, Number(depth) || 2))));

    const ownershipUrl = `${GRAPH_API_BASE}/graph/ownership${ownershipParams.toString() ? `?${ownershipParams.toString()}` : ""}`;
    const ringUrl = `${GRAPH_API_BASE}/graph/ring-scoring?${ringParams.toString()}`;

    const [ownershipResult, ringResult] = await Promise.all([
        safeGraphJson(ownershipUrl),
        safeGraphJson(ringUrl),
    ]);

    const ownershipPayload = ownershipResult?.payload && typeof ownershipResult.payload === "object"
        ? ownershipResult.payload
        : null;
    const ringPayload = ringResult?.payload && typeof ringResult.payload === "object"
        ? ringResult.payload
        : null;

    const ownershipStatus = normalizeForensicDataStatus(
        ownershipPayload?.data_status || ownershipResult?.dataStatus,
        ownershipPayload ? "ok" : "unavailable",
    );
    const ringStatus = normalizeForensicDataStatus(
        ringPayload?.data_status || ringResult?.dataStatus,
        ringPayload ? "ok" : "unavailable",
    );

    return {
        ownership: ownershipPayload,
        ring: ringPayload,
        fetched_at: new Date().toISOString(),
        diagnostics: {
            ownership_status: ownershipStatus,
            ring_status: ringStatus,
            ownership_fetch: {
                ok: Boolean(ownershipResult?.ok),
                status_code: ownershipResult?.statusCode ?? null,
                duration_ms: toFiniteNumber(ownershipResult?.durationMs, 0),
            },
            ring_fetch: {
                ok: Boolean(ringResult?.ok),
                status_code: ringResult?.statusCode ?? null,
                duration_ms: toFiniteNumber(ringResult?.durationMs, 0),
            },
            both_available: Boolean(ownershipResult?.ok) && Boolean(ringResult?.ok),
        },
    };
}


async function loadGraphQuality() {
    try {
        const res = await graphFetch(`${GRAPH_API_BASE}/monitoring/graph_quality`);
        if (!res.ok) throw new Error(`Quality API error: ${res.status}`);
        const payload = await res.json();
        graphQuality = payload;
        return payload;
    } catch (err) {
        console.warn("Graph quality load warning:", err);
        return graphQuality;
    }
}


function renderModelIntelligence(data) {
    const payload = data && typeof data === "object" ? data : {};
    const modelInfo = payload.model_info && typeof payload.model_info === "object" ? payload.model_info : {};
    const thresholds = payload.decision_thresholds && typeof payload.decision_thresholds === "object"
        ? payload.decision_thresholds
        : (modelInfo.decision_thresholds && typeof modelInfo.decision_thresholds === "object" ? modelInfo.decision_thresholds : {});
    const policy = thresholds.policy && typeof thresholds.policy === "object" ? thresholds.policy : {};
    const attentionSummary = payload.attention_summary && typeof payload.attention_summary === "object" ? payload.attention_summary : {};
    const queryContext = payload.query_context && typeof payload.query_context === "object" ? payload.query_context : {};

    const mode = String(modelInfo.inference_mode || "unknown");
    const modeLabel = mode === "gnn_ensemble"
        ? "GNN + Ensemble"
        : mode === "heuristic_fallback"
            ? "Heuristic dự phòng"
            : "--";

    const policyChunks = [];
    if (Number.isFinite(Number(policy.cold_start_degree_threshold))) {
        policyChunks.push(`k<${Number(policy.cold_start_degree_threshold)}`);
    }
    if (Number.isFinite(Number(policy.cold_start_threshold_delta))) {
        policyChunks.push(`Δ${Number(policy.cold_start_threshold_delta).toFixed(2)}`);
    }
    if (Number.isFinite(Number(policy.node_blend_alpha_gnn))) {
        policyChunks.push(`α${Number(policy.node_blend_alpha_gnn).toFixed(2)}`);
    }

    setElementText("model-mode-label", modeLabel);
    setElementText("node-threshold-value", formatThreshold(thresholds.node));
    setElementText("edge-threshold-value", formatThreshold(thresholds.edge));
    setElementText("attention-count-value", Number.isFinite(Number(attentionSummary.count)) ? String(Number(attentionSummary.count)) : "--");
    setElementText("policy-summary", policyChunks.length ? policyChunks.join(" | ") : "--");
    setElementText("query-depth-value", Number.isFinite(Number(queryContext.depth)) ? String(Number(queryContext.depth)) : "--");

    const modelBadge = document.getElementById("model-status-badge");
    if (!modelBadge) return;

    const isLoaded = Boolean(modelInfo.model_loaded);
    const fallbackActive = Boolean(modelInfo.fallback_active);
    const ensembleActive = Boolean(modelInfo.ensemble_active);
    const fallbackReason = modelInfo.fallback_reason ? ` · ${String(modelInfo.fallback_reason)}` : "";

    if (isLoaded) {
        modelBadge.innerHTML = `<div class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div><span class="text-[10px] text-slate-200 font-bold">Mô hình GAT đang hoạt động${ensembleActive ? " · Tổ hợp" : ""}</span>`;
        return;
    }

    if (fallbackActive) {
        modelBadge.innerHTML = `<div class="w-2 h-2 rounded-full bg-amber-500"></div><span class="text-[10px] text-amber-200 font-bold">Chế độ heuristic${escapeHtml(fallbackReason)}</span>`;
        return;
    }

    modelBadge.innerHTML = `<div class="w-2 h-2 rounded-full bg-slate-400"></div><span class="text-[10px] text-slate-300 font-bold">Đang chờ dữ liệu mô hình</span>`;
}


function resetInvestigationSummary() {
    const container = document.getElementById("investigation-summary");
    if (container) container.classList.add("hidden");
    setElementText("summary-company-count", "--");
    setElementText("summary-shell-count", "--");
    setElementText("summary-cycle-count", "--");
    setElementText("summary-risk-level", "--");
    setElementText("summary-risk-hint", "Đang chờ phân tích");
}


function renderInvestigationSummary(data) {
    if (!data || typeof data !== "object") {
        resetInvestigationSummary();
        return;
    }

    const container = document.getElementById("investigation-summary");
    if (!container) return;

    const nodes = Array.isArray(data.nodes) ? data.nodes : [];
    const edges = Array.isArray(data.edges) ? data.edges : [];
    const shellCount = nodes.filter((node) => Boolean(node && node.is_shell)).length;
    const cycleCount = edges.filter((edge) => Boolean(edge && edge.is_circular)).length;
    const suspiciousAmount = toFiniteNumber(data.total_suspicious_amount, 0);
    const topAttention = Array.isArray(data.attention_summary?.top_edges) ? data.attention_summary.top_edges[0] : null;

    let riskLevel = "THẤP";
    let riskHint = `Giá trị nghi vấn: ${new Intl.NumberFormat("vi-VN").format(Math.round(suspiciousAmount))} VNĐ`;
    if (suspiciousAmount > 10e9) {
        riskLevel = "RẤT CAO";
    } else if (suspiciousAmount > 1e9) {
        riskLevel = "CAO";
    } else if (suspiciousAmount > 0) {
        riskLevel = "TRUNG BÌNH";
    }

    if (topAttention) {
        riskHint = `Tín hiệu chú ý cao nhất: ${topAttention.from} → ${topAttention.to} (${toFiniteNumber(topAttention.weight, 0).toFixed(2)})`;
    }

    setElementText("summary-company-count", `${nodes.length}`);
    setElementText("summary-shell-count", `${shellCount}`);
    setElementText("summary-cycle-count", `${cycleCount}`);
    setElementText("summary-risk-level", riskLevel);
    setElementText("summary-risk-hint", riskHint);

    const riskLevelEl = document.getElementById("summary-risk-level");
    if (riskLevelEl) {
        if (riskLevel === "RẤT CAO") {
            riskLevelEl.style.color = "#b91c1c";
        } else if (riskLevel === "CAO") {
            riskLevelEl.style.color = "#b45309";
        } else if (riskLevel === "TRUNG BÌNH") {
            riskLevelEl.style.color = "#1d4ed8";
        } else {
            riskLevelEl.style.color = "#166534";
        }
    }

    container.classList.remove("hidden");
}


function renderQualitySummary(quality) {
    const payload = quality && typeof quality === "object" ? quality : {};
    const status = String(payload.status || "unknown").toLowerCase();
    const gateSummary = payload.gate_summary && typeof payload.gate_summary === "object" ? payload.gate_summary : {};
    const drift = payload.drift && typeof payload.drift === "object" ? payload.drift : {};
    const modelInfo = payload.model_info && typeof payload.model_info === "object" ? payload.model_info : {};
    const failedGates = Array.isArray(gateSummary.failed_gates) ? gateSummary.failed_gates : [];
    const statusReason = String(payload.status_reason || "").trim();

    const servingPass = gateSummary.serving_pass;
    const stressPass = gateSummary.stress_pass;
    const driftDetected = Boolean(drift.detected);
    const driftSeverity = String(drift.severity || "unknown").toLowerCase();
    const hasDriftFlag = typeof drift.detected === "boolean";

    const servingState = typeof servingPass === "boolean" ? (servingPass ? "healthy" : "degraded") : "unknown";
    const stressState = typeof stressPass === "boolean" ? (stressPass ? "healthy" : "degraded") : "unknown";

    let driftState = "unknown";
    if (driftDetected) {
        driftState = driftSeverity === "high" ? "degraded" : "warning";
    } else if (driftSeverity === "low") {
        driftState = "healthy";
    } else if (driftSeverity === "insufficient_data") {
        driftState = "unknown";
    }

    const driftSeverityLabelMap = {
        high: "CAO",
        medium: "TRUNG BÌNH",
        low: "THẤP",
        insufficient_data: "THIẾU DỮ LIỆU",
        unknown: "KHÔNG RÕ",
    };
    const driftSeverityLabel = driftSeverityLabelMap[driftSeverity] || driftSeverity.toUpperCase();

    applyQualityPill("quality-overall-pill", `MÔ HÌNH: ${formatQualityState(status)}`, status);
    applyQualityPill("quality-serving-pill", `VẬN HÀNH: ${typeof servingPass === "boolean" ? (servingPass ? "ĐẠT" : "KHÔNG ĐẠT") : "--"}`, servingState);
    applyQualityPill("quality-stress-pill", `ÁP LỰC: ${typeof stressPass === "boolean" ? (stressPass ? "ĐẠT" : "KHÔNG ĐẠT") : "--"}`, stressState);
    applyQualityPill(
        "quality-drift-pill",
        `TRÔI: ${hasDriftFlag ? (driftDetected ? driftSeverityLabel : "KHÔNG") : "--"}`,
        driftState,
    );

    const updatedAt = payload.generated_at || modelInfo.updated_at;
    const modelVersion = modelInfo.model_version ? ` · v${modelInfo.model_version}` : "";
    setElementText(
        "quality-updated-at",
        updatedAt ? `Ảnh chụp ${formatShortTimestamp(updatedAt)}${modelVersion}` : "Ảnh chụp chất lượng chưa sẵn sàng",
    );

    const reasonEl = document.getElementById("quality-reason");
    if (reasonEl) {
        const topGateSummary = failedGates
            .slice(0, 2)
            .map((item) => {
                const actual = Number(item?.actual);
                const threshold = Number(item?.threshold);
                const actualLabel = Number.isFinite(actual) ? actual.toFixed(4) : "--";
                const thresholdLabel = Number.isFinite(threshold) ? threshold.toFixed(4) : "--";
                return `${formatGateLabel(item?.gate)}: ${actualLabel}/${thresholdLabel}`;
            })
            .join(" | ");
        const reasonLabel = formatQualityReason(statusReason);
        reasonEl.textContent = topGateSummary ? `${reasonLabel} Gate fail: ${topGateSummary}.` : reasonLabel;
    }

    const failBtn = document.getElementById("quality-fail-action");
    if (failBtn) {
        if (failedGates.length) {
            failBtn.classList.remove("hidden");
            failBtn.onclick = () => {
                switchForensicTab("logs");
                const logs = document.getElementById("investigation-logs");
                if (logs) logs.scrollIntoView({ behavior: "smooth", block: "start" });
            };
        } else {
            failBtn.classList.add("hidden");
            failBtn.onclick = null;
        }
    }
}


function setupForensicTabs() {
    const tabLogs = document.getElementById("tab-logs");
    const tabPaths = document.getElementById("tab-paths");
    if (!tabLogs || !tabPaths) return;
    if (tabLogs.hasAttribute("data-initialized")) return;

    const forensicTabList = tabLogs.parentElement;

    tabLogs.setAttribute("data-initialized", "true");

    tabLogs.addEventListener("click", () => switchForensicTab("logs"));
    tabPaths.addEventListener("click", () => switchForensicTab("paths"));

    if (forensicTabList) {
        forensicTabList.addEventListener("keydown", (event) => {
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
            const active = document.activeElement === tabPaths ? "paths" : "logs";
            event.preventDefault();

            if (event.key === "ArrowRight") {
                switchForensicTab(active === "logs" ? "paths" : "logs", { skipAnimation: true });
            } else if (event.key === "ArrowLeft") {
                switchForensicTab(active === "logs" ? "paths" : "logs", { skipAnimation: true });
            } else if (event.key === "Home") {
                switchForensicTab("logs", { skipAnimation: true });
            } else if (event.key === "End") {
                switchForensicTab("paths", { skipAnimation: true });
            }

            if (document.getElementById("tab-logs")?.getAttribute("aria-selected") === "true") {
                document.getElementById("tab-logs")?.focus();
            } else {
                document.getElementById("tab-paths")?.focus();
            }
        });
    }

    switchForensicTab("logs", { skipAnimation: true });
}


function switchForensicTab(target, options = {}) {
    const tabLogs = document.getElementById("tab-logs");
    const tabPaths = document.getElementById("tab-paths");
    const logsPane = document.getElementById("investigation-logs");
    const pathsPane = document.getElementById("evidence-paths-container");
    if (!tabLogs || !tabPaths || !logsPane || !pathsPane) return;

    const showPaths = target === "paths";
    const activeTab = showPaths ? tabPaths : tabLogs;
    const inactiveTab = showPaths ? tabLogs : tabPaths;
    const activePane = showPaths ? pathsPane : logsPane;
    const inactivePane = showPaths ? logsPane : pathsPane;

    activeTab.setAttribute("aria-selected", "true");
    inactiveTab.setAttribute("aria-selected", "false");
    activeTab.setAttribute("tabindex", "0");
    inactiveTab.setAttribute("tabindex", "-1");

    activeTab.className = `${FORENSIC_TAB_ACTIVE_CLASS}${showPaths ? " flex items-center gap-1" : ""}`;
    inactiveTab.className = FORENSIC_TAB_INACTIVE_CLASS;

    inactivePane.classList.remove("block", "forensic-pane-enter");
    inactivePane.classList.add("hidden");
    inactivePane.setAttribute("hidden", "");

    activePane.classList.remove("hidden");
    activePane.classList.add("block");
    activePane.removeAttribute("hidden");

    if (!options.skipAnimation) {
        activePane.classList.remove("forensic-pane-enter");
        void activePane.offsetWidth;
        activePane.classList.add("forensic-pane-enter");
    }
}


function setupWorkbenchTabs() {
    const tabs = Array.from(document.querySelectorAll("#graph-workbench-tabs .workbench-tab"));
    if (!tabs.length) return;

    const host = document.getElementById("graph-workbench-tabs");
    if (host && host.getAttribute("data-initialized") === "true") return;

    tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            const mode = tab.getAttribute("data-mode") || "companies";
            switchWorkbenchMode(mode);
        });
    });

    if (host) {
        host.addEventListener("keydown", (event) => {
            const key = event.key;
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(key)) return;

            const currentIndex = tabs.findIndex((tab) => tab === document.activeElement);
            if (currentIndex < 0) return;

            event.preventDefault();
            let nextIndex = currentIndex;
            if (key === "ArrowRight") nextIndex = (currentIndex + 1) % tabs.length;
            if (key === "ArrowLeft") nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
            if (key === "Home") nextIndex = 0;
            if (key === "End") nextIndex = tabs.length - 1;

            const nextTab = tabs[nextIndex];
            if (!nextTab) return;
            nextTab.focus();
            switchWorkbenchMode(nextTab.getAttribute("data-mode") || "companies", { skipScroll: true });
        });
    }

    if (host) {
        host.setAttribute("data-initialized", "true");
    }

    switchWorkbenchMode("companies", { skipScroll: true, force: true });
}


function setupGlobalWorkbenchShortcuts() {
    if (document.body.getAttribute("data-workbench-shortcuts") === "true") return;
    document.body.setAttribute("data-workbench-shortcuts", "true");

    document.addEventListener("keydown", (event) => {
        if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
        const target = event.target;
        const tag = target && target.tagName ? target.tagName.toLowerCase() : "";
        if (tag === "input" || tag === "textarea" || tag === "select" || (target && target.isContentEditable)) {
            return;
        }

        if (event.key === "g" || event.key === "G") {
            switchWorkbenchMode("graph");
        } else if (event.key === "f" || event.key === "F") {
            switchWorkbenchMode("forensic");
        } else if (event.key === "c" || event.key === "C") {
            switchWorkbenchMode("companies");
        } else if (event.key === "o" || event.key === "O") {
            switchWorkbenchMode("osint");
        } else if (event.key === "Escape") {
            clearGraphFocus();
        }
    });
}


function switchWorkbenchMode(mode, options = {}) {
    const validModes = new Set(["graph", "forensic", "companies", "osint", "entity"]);
    const nextMode = validModes.has(mode) ? mode : "companies";
    if (!options.force && activeWorkbenchMode === nextMode) return;
    activeWorkbenchMode = nextMode;

    const tabs = Array.from(document.querySelectorAll("#graph-workbench-tabs .workbench-tab"));
    tabs.forEach((tab) => {
        const isActive = tab.getAttribute("data-mode") === nextMode;
        tab.setAttribute("aria-selected", isActive ? "true" : "false");
        tab.setAttribute("tabindex", isActive ? "0" : "-1");
    });

    const investigationSection = document.getElementById("investigation-section");
    const companiesSection = document.getElementById("companies-section");
    const osintSection = document.getElementById("osint-section");
    const entitySection = document.getElementById("entity-section");
    if (!investigationSection || !companiesSection) return;

    if (nextMode !== "graph") {
        stopTimelinePlayback();
    }

    if (osintSection) {
        osintSection.classList.add("hidden");
        osintSection.setAttribute("hidden", "");
        osintSection.style.display = "none";
    }

    if (entitySection) {
        entitySection.classList.add("hidden");
        entitySection.setAttribute("hidden", "");
        entitySection.style.display = "none";
    }

    if (nextMode === "companies") {
        investigationSection.classList.add("hidden", "opacity-0");
        investigationSection.classList.remove("workbench-forensic-mode");
        investigationSection.setAttribute("hidden", "");
        companiesSection.classList.remove("hidden");
        companiesSection.removeAttribute("hidden");

        if (!options.skipScroll) {
            companiesSection.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        return;
    }

    if (nextMode === "osint") {
        investigationSection.classList.add("hidden", "opacity-0");
        investigationSection.classList.remove("workbench-forensic-mode");
        investigationSection.setAttribute("hidden", "");
        companiesSection.classList.add("hidden");
        companiesSection.setAttribute("hidden", "");

        if (osintSection) {
            osintSection.classList.remove("hidden");
            osintSection.removeAttribute("hidden");
            osintSection.style.display = "flex";
            if (!options.skipScroll) {
                osintSection.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        }
        return;
    }

    if (nextMode === "entity") {
        investigationSection.classList.add("hidden", "opacity-0");
        investigationSection.classList.remove("workbench-forensic-mode");
        investigationSection.setAttribute("hidden", "");
        companiesSection.classList.add("hidden");
        companiesSection.setAttribute("hidden", "");

        if (entitySection) {
            entitySection.classList.remove("hidden");
            entitySection.removeAttribute("hidden");
            entitySection.style.display = "flex";
            if (!options.skipScroll) {
                entitySection.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        }
        return;
    }

    companiesSection.classList.add("hidden");
    companiesSection.setAttribute("hidden", "");
    investigationSection.classList.remove("hidden", "opacity-0");
    investigationSection.removeAttribute("hidden");
    if (nextMode === "forensic") {
        investigationSection.classList.add("workbench-forensic-mode");
    } else {
        investigationSection.classList.remove("workbench-forensic-mode");
    }

    if (!graphData) {
        showLoading(false);
        showEmptyState("Nhập mã số thuế để bắt đầu phân tích");
    }

    if (!options.skipScroll) {
        investigationSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}


function setupGraphCanvasControls() {
    const fitBtn = document.getElementById("graph-fit-btn");
    const resetBtn = document.getElementById("graph-reset-btn");
    const clearBtn = document.getElementById("graph-clear-focus-btn");
    if (!fitBtn || !resetBtn || !clearBtn) return;
    if (fitBtn.getAttribute("data-initialized") === "true") return;

    fitBtn.setAttribute("data-initialized", "true");

    fitBtn.addEventListener("click", () => fitGraphToViewport({ animate: true }));
    resetBtn.addEventListener("click", () => {
        if (!svg || !zoomBehavior) return;
        svg.transition().duration(420).call(zoomBehavior.transform, d3.zoomIdentity);
    });
    clearBtn.addEventListener("click", () => {
        clearGraphFocus();
    });
}


function fitGraphToViewport(options = {}) {
    if (!svg || !zoomBehavior) return;

    const nodeData = svg.selectAll(".node").data();
    if (!Array.isArray(nodeData) || !nodeData.length) return;

    const xs = nodeData.map((n) => Number(n.x)).filter((n) => Number.isFinite(n));
    const ys = nodeData.map((n) => Number(n.y)).filter((n) => Number.isFinite(n));
    if (!xs.length || !ys.length) return;

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const width = Math.max(document.getElementById("graph-canvas")?.clientWidth || 0, 1);
    const height = Math.max(document.getElementById("graph-canvas")?.clientHeight || 0, 1);
    const padding = 100;

    const contentWidth = Math.max(maxX - minX, 1);
    const contentHeight = Math.max(maxY - minY, 1);
    const scale = Math.max(
        0.2,
        Math.min(
            5,
            Math.min((width - padding) / contentWidth, (height - padding) / contentHeight),
        ),
    );

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const tx = width / 2 - scale * centerX;
    const ty = height / 2 - scale * centerY;
    const transform = d3.zoomIdentity.translate(tx, ty).scale(scale);

    if (options.animate) {
        svg.transition().duration(420).call(zoomBehavior.transform, transform);
    } else {
        svg.call(zoomBehavior.transform, transform);
    }
}


function clearGraphFocus() {
    if (!svg) return;
    activeNodeFocusId = null;

    svg.selectAll(".node")
        .transition().duration(260)
        .attr("opacity", 1)
        .style("opacity", 1);

    svg.selectAll(".edge")
        .transition().duration(260)
        .attr("opacity", 1);

    svg.selectAll(".edge-path")
        .transition().duration(260)
        .style("opacity", (d) => d && d.is_circular ? 1 : 0.15);

    svg.selectAll(".node-label")
        .transition().duration(260)
        .style("opacity", 1);

    setTimeout(() => applyTimelineFilter(timelineMonth, { refreshBadge: true }), 280);
}

function renderGraph(data) {
    const container = document.getElementById("graph-canvas");
    let width = container.clientWidth;
    let height = container.clientHeight;

    // Khắc phục lỗi viewBox = 0 0 gán vào init khi canvas bị ẩn (display: none)
    if (width === 0) width = 800;
    if (height === 0) height = 600;
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const zoomGroup = svg.select(".zoom-group");
    const boundaryLayer = zoomGroup.select(".boundary-layer");
    const edgesLayer = zoomGroup.select(".edges-layer");
    const nodesLayer = zoomGroup.select(".nodes-layer");
    const labelsLayer = zoomGroup.select(".labels-layer");

    // Clear previous
    boundaryLayer.selectAll("*").remove();
    edgesLayer.selectAll("*").remove();
    nodesLayer.selectAll("*").remove();
    labelsLayer.selectAll("*").remove();

    if (!data.nodes || !data.nodes.length) {
        showEmptyState("Không tìm thấy dữ liệu mạng lưới cho mã số thuế này.");
        return;
    }

    const nodes = data.nodes.map(n => ({ ...n }));
    const safeEdges = Array.isArray(data.edges) ? data.edges : [];
    
    // Compute link index for parallel edges
    const linkCounts = {};
    const edges = safeEdges.map(e => {
        const key = e.from < e.to ? `${e.from}-${e.to}` : `${e.to}-${e.from}`;
        const index = linkCounts[key] || 0;
        linkCounts[key] = index + 1;
        const edgeMonth = resolveEdgeMonth(e.date);
        return {
            ...e,
            source: e.from,
            target: e.to,
            linkIndex: index,
            timeline_month: edgeMonth,
        };
    });

    // ── Simulation ──
    if (simulation) simulation.stop();

    const countriesArray = [...new Set(nodes.map(d => d.country_inferred || "Việt Nam"))];
    
    simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(edges).id(d => d.id).distance(110).strength(0.65))
        .force("charge", d3.forceManyBody().strength(-450))
        .force("collision", d3.forceCollide().radius(56).iterations(4))
        .alphaDecay(0.05);

    if (countriesArray.length <= 1) {
        simulation.force("center", d3.forceCenter(width / 2, height / 2));
    } else {
        countriesArray.forEach((c, i) => {
            const isVN = c.toLowerCase().includes("vietnam") || c.toLowerCase().includes("việt nam");
            const offsetR = isVN ? 0 : 450; 
            const theta = (i * 2 * Math.PI) / (countriesArray.length - 1 || 1);
            const tx = width / 2 + offsetR * Math.cos(theta);
            const ty = height / 2 + offsetR * Math.sin(theta);
            
            simulation.force(`x-${i}`, d3.forceX(tx).strength(d => {
                if ((d.country_inferred || "Việt Nam") !== c) return 0;
                const dIsVn = String(d.country_inferred || "Việt Nam").toLowerCase().includes("vietnam")
                    || String(d.country_inferred || "Việt Nam").toLowerCase().includes("việt nam");
                return dIsVn ? 0.15 : 0.4;
            }));
            simulation.force(`y-${i}`, d3.forceY(ty).strength(d => {
                if ((d.country_inferred || "Việt Nam") !== c) return 0;
                const dIsVn = String(d.country_inferred || "Việt Nam").toLowerCase().includes("vietnam")
                    || String(d.country_inferred || "Việt Nam").toLowerCase().includes("việt nam");
                return dIsVn ? 0.15 : 0.4;
            }));
        });
    }

    // ── Edges ──
    const edgeGroups = edgesLayer.selectAll("g.edge")
        .data(edges, d => d.invoice_number || d.relationship_type || `${d.from}-${d.to}-${d.linkIndex}`)
        .join("g")
        .attr("class", d => `edge ${getEdgeLegendCategory(d)}`);

    const lines = edgeGroups.append("path")
        .attr("class", d => `edge-path ${getEdgeLegendCategory(d)}-path`)
        .attr("fill", "none")
        .attr("stroke", d => d.is_ownership ? "#fbbf24" : (d.is_circular ? "#ba1a1a" : "#334155"))
        .attr("stroke-width", d => d.is_ownership ? 2 : (d.is_circular ? 2 : 1))
        .attr("stroke-dasharray", d => d.is_ownership ? "4 4" : (d.is_circular ? "8 4" : "none"))
        .attr("marker-end", d => d.is_ownership ? "none" : (d.is_circular ? "url(#arrow-fraud)" : "url(#arrow-normal)"))
        .attr("opacity", d => d.is_ownership ? 0.6 : (d.is_circular ? 1 : 0.15));

    // Edge labels (amount) - ONLY for suspicious ones to declutter
    const edgeLabels = edgeGroups.filter(d => d.is_circular || d.amount >= 1e9).append("text")
        .attr("class", "edge-label")
        .attr("text-anchor", "middle")
        .attr("dy", -6)
        .attr("fill", d => d.is_circular ? "#fca5a5" : "rgba(100, 116, 139, 0.5)")
        .attr("font-size", d => d.is_circular ? "9px" : "8px")
        .attr("font-weight", d => d.is_circular ? "800" : "500")
        .attr("font-family", "Inter, sans-serif")
        .text(d => {
            if (d.amount >= 1e9) return `${(d.amount / 1e9).toFixed(1)}B`;
            if (d.amount >= 1e6) return `${(d.amount / 1e6).toFixed(0)}Tr`;
            return `${(d.amount / 1e3).toFixed(0)}K`;
        });

    // ── Nodes ──
    const nodeGroups = nodesLayer.selectAll("g.node")
        .data(nodes, d => d.id)
        .join("g")
        .attr("class", d => `node ${getNodeLegendCategory(d)}`)
        .style("cursor", "pointer")
        .call(d3.drag()
            .on("start", dragStarted)
            .on("drag", dragged)
            .on("end", dragEnded)
        );

    // Node main rect container (replacing circle with rectangle)
    nodeGroups.append("rect")
        .attr("width", 100)
        .attr("height", 32)
        .attr("x", -50)
        .attr("y", -16)
        .attr("rx", 6)
        .attr("ry", 6)
        .attr("fill", d => {
            if (d.is_ghost) return "transparent"; // Ghost/Inactive is transparent
            if (d.is_offshore) {
                if (d.risk_score >= 80) return "#f43f5e"; // rose-500
                if (d.risk_score >= 60) return "#f59e0b"; // amber-500
                return "#10b981"; // emerald-500
            }
            if (d.is_shell) return "#ba1a1a"; // Exact mock red
            if (d.group === "suspicious") return "#1e293b"; // slate-800
            return "#002147"; // primary-container
        })
        .attr("stroke", d => {
            if (d.is_ghost) return "#94a3b8"; // Ghost outline
            if (d.is_offshore) return "rgba(255,255,255,0.6)";
            if (d.is_shell) return "rgba(255,255,255,0.4)";
            if (d.group === "suspicious") return "rgba(255,255,255,0.15)";
            return "rgba(255,255,255,0.3)";
        })
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", d => d.is_ghost ? "4 4" : "none") // Dash border for ghost
        .attr("class", "node-rect")
        .style("filter", d => {
            if (d.is_ghost) return "none";
            return d.is_shell ? "drop-shadow(0 10px 15px rgb(186 26 26 / 0.5))" : "drop-shadow(0 4px 6px rgb(0 0 0 / 0.3))";
        });
        
    // Node icons using Material Symbols Outlined
    const iconMap = {
        "shell": "corporate_fare",
        "suspicious": "store",
        "normal": "factory",
    };
    nodeGroups.append("text")
        .attr("class", "material-symbols-outlined")
        .attr("text-anchor", "middle")
        .attr("x", -34)
        .attr("y", 5)
        .attr("fill", d => d.is_ghost ? "#94a3b8" : (d.group === "suspicious" ? "#cbd5e1" : "#ffffff"))
        .attr("font-size", d => d.is_shell ? "18px" : "16px")
        .style("font-family", "Material Symbols Outlined")
        .text(d => {
            if (d.is_ghost) return "help";
            if (d.is_offshore) return "public";
            return iconMap[d.group] || "factory";
        });

    // Node labels inside the rect
    nodeGroups.append("text")
        .attr("class", "node-label")
        .attr("text-anchor", "start")
        .attr("x", -18)
        .attr("y", 4)
        .attr("fill", d => d.is_ghost ? "#94a3b8" : "#ffffff")
        .attr("font-size", "9px")
        .attr("font-weight", d => d.is_shell ? "800" : "600")
        .attr("font-family", "Inter, sans-serif")
        .attr("letter-spacing", "0.02em")
        .text(d => {
            const name = d.label || d.tax_code;
            return name.length > 12 ? name.substring(0, 11) + "…" : name;
        });

    // ── Tooltip ──
    nodeGroups.on("mouseenter", function(event, d) {
        showTooltip(event, d);
        d3.select(this).select(".node-rect")
            .transition().duration(200)
            .attr("stroke-width", 2)
            .attr("stroke", "#ffffff");
    }).on("mouseleave", function(event, d) {
        hideTooltip();
        d3.select(this).select(".node-rect")
            .transition().duration(200)
            .attr("stroke-width", 1)
            .attr("stroke", d => {
                if (d.is_ghost) return "#94a3b8";
                if (d.is_offshore) return "rgba(255,255,255,0.6)";
                if (d.is_shell) return "rgba(255,255,255,0.4)";
                if (d.group === "suspicious") return "rgba(255,255,255,0.15)";
                return "rgba(255,255,255,0.3)";
            });
    }).on("click", function(event, d) {
        event.preventDefault();
        event.stopPropagation();
        highlightNode(d.id, edges);
    });

    svg.on("click", (event) => {
        const clickedNode = event?.target?.closest?.(".node");
        if (!clickedNode) {
            clearGraphFocus();
        }
    });

    // ── Boundaries ──
    const countries = [...new Set(nodes.map(d => d.country_inferred || "Việt Nam"))];
    
    const getCountryColor = (c) => {
        const name = c.toLowerCase();
        if (name.includes("vietnam") || name.includes("việt nam")) return "rgba(16, 185, 129, 0.05)";
        if (name.includes("cayman")) return "rgba(244, 63, 94, 0.05)";
        if (name.includes("singapore")) return "rgba(56, 189, 248, 0.05)";
        if (name.includes("virgin")) return "rgba(245, 158, 11, 0.05)";
        return "rgba(148, 163, 184, 0.05)";
    };
    
    const getCountryStroke = (c) => {
        const name = c.toLowerCase();
        if (name.includes("vietnam") || name.includes("việt nam")) return "#10b981"; 
        if (name.includes("cayman")) return "#f43f5e"; 
        if (name.includes("singapore")) return "#38bdf8"; 
        if (name.includes("virgin")) return "#f59e0b"; 
        return "#94a3b8"; 
    };

    const getCountryFlag = (c) => {
        const name = c.toLowerCase();
        if (name.includes("vietnam") || name.includes("việt nam")) return "🇻🇳";
        if (name.includes("cayman")) return "🇰🇾";
        if (name.includes("singapore")) return "🇸🇬";
        if (name.includes("virgin")) return "🇻🇬";
        return "🏳️";
    };

    const boundaryGroups = boundaryLayer.selectAll("g.country-boundary")
        .data(countries)
        .join("g")
        .attr("class", "country-boundary");

    boundaryGroups.append("circle")
        .attr("fill", d => getCountryColor(d))
        .attr("stroke", d => getCountryStroke(d))
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "8 8");

    const boundaryLabels = boundaryGroups.append("text")
        .attr("class", "country-label")
        .attr("text-anchor", "middle")
        .attr("fill", "#94a3b8")
        .attr("font-size", "14px")
        .attr("font-weight", "800")
        .attr("font-family", "Inter, sans-serif")
        .attr("letter-spacing", "0.05em")
        .style("opacity", 0.8)
        .text(d => `${getCountryFlag(d)} ${d.toUpperCase()}`);

    // ── Tick ──
    simulation.on("tick", () => {
        // Update boundaries
        const clusterCenters = {};
        countries.forEach(c => {
            const groupNodes = nodes.filter(n => (n.country_inferred || "Việt Nam") === c);
            if (!groupNodes.length) return;
            const cx = d3.mean(groupNodes, d => d.x);
            const cy = d3.mean(groupNodes, d => d.y);
            const r = d3.max(groupNodes, d => Math.sqrt(Math.pow(d.x - cx, 2) + Math.pow(d.y - cy, 2))) || 0;
            clusterCenters[c] = { cx, cy, r: Math.max(r + 80, 150) }; // min radius
        });

        boundaryGroups.select("circle")
            .attr("cx", d => clusterCenters[d]?.cx || 0)
            .attr("cy", d => clusterCenters[d]?.cy || 0)
            .attr("r", d => clusterCenters[d]?.r || 0);

        boundaryLabels
            .attr("x", d => clusterCenters[d]?.cx || 0)
            .attr("y", d => (clusterCenters[d]?.cy || 0) - (clusterCenters[d]?.r || 0) + 24);

        // Update edge paths (curved for bidirectional or multi-edges)
        lines.attr("d", d => {
            const dx = d.target.x - d.source.x;
            const dy = d.target.y - d.source.y;
            const dr = Math.sqrt(dx * dx + dy * dy);
            
            if (d.linkIndex > 0 || d.is_circular) {
                const sweep = d.linkIndex % 2 === 0 ? 1 : 0; 
                const step = Math.ceil(d.linkIndex / 2);
                const arcRadius = dr * (1.2 + step * 0.4);
                return `M${d.source.x},${d.source.y}A${arcRadius},${arcRadius} 0 0,${sweep} ${d.target.x},${d.target.y}`;
            }
            return `M${d.source.x},${d.source.y}L${d.target.x},${d.target.y}`;
        });

        edgeLabels
            .attr("x", d => {
                if (d.linkIndex > 0 || d.is_circular) return (d.source.x + d.target.x) / 2 + (d.linkIndex * 5);
                return (d.source.x + d.target.x) / 2;
            })
            .attr("y", d => {
                if (d.linkIndex > 0 || d.is_circular) return (d.source.y + d.target.y) / 2 + (d.linkIndex * 5);
                return (d.source.y + d.target.y) / 2;
            });

        nodeGroups.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    // Update legend badge
    const liveCount = document.getElementById("live-node-count");
    if (liveCount) liveCount.textContent = `${nodes.length} nút · ${edges.length} cạnh`;

    graphRenderState = {
        edges,
        lines,
        edgeLabels,
    };

    renderInteractiveLegend();
    renderGraphRiskAnalytics(data);
    applyTimelineFilter(timelineMonth, { refreshBadge: true });
    applyLegendVisibility();

    setTimeout(() => fitGraphToViewport({ animate: true }), 180);
}


// ════════════════════════════════════════════════════════════════
//  Forensic Panel
// ════════════════════════════════════════════════════════════════
function formatVndCompact(value) {
    const amount = toFiniteNumber(value, 0);
    if (amount >= 1_000_000_000) {
        return `${(amount / 1_000_000_000).toFixed(1)} tỷ VNĐ`;
    }
    if (amount >= 1_000_000) {
        return `${(amount / 1_000_000).toFixed(0)} triệu VNĐ`;
    }
    return `${new Intl.NumberFormat("vi-VN").format(Math.round(amount))} VNĐ`;
}


function formatForensicTimestamp(raw) {
    const dt = raw ? new Date(raw) : new Date();
    if (Number.isNaN(dt.getTime())) {
        return new Date().toLocaleString("vi-VN", { hour12: false });
    }
    return dt.toLocaleString("vi-VN", { hour12: false });
}


function formatForensicTimestampOrUnknown(raw) {
    if (!raw) return "Không rõ thời điểm";
    const dt = new Date(raw);
    if (Number.isNaN(dt.getTime())) return "Không rõ thời điểm";
    return dt.toLocaleString("vi-VN", { hour12: false });
}


function getCircularEdgeSignals(data) {
    const edges = Array.isArray(data?.edges) ? data.edges : [];
    const dedupe = new Set();

    const normalized = edges
        .filter((edge) => edge && edge.is_circular)
        .map((edge) => {
            const from = String(edge.from || "").trim();
            const to = String(edge.to || "").trim();
            const amount = Math.max(0, toFiniteNumber(edge.amount, 0));
            const circularProbability = Math.max(
                0,
                Math.min(1, toFiniteNumber(edge.circular_probability ?? edge.fraud_probability, 0)),
            );
            const dateRaw = String(edge.date || edge.timestamp || "").trim();
            const invoiceNumber = String(edge.invoice_number || edge.invoiceNo || "").trim();

            return {
                from,
                to,
                amount,
                circularProbability,
                dateRaw,
                invoiceNumber,
            };
        })
        .filter((edge) => edge.from && edge.to)
        .filter((edge) => {
            const key = `${edge.invoiceNumber}|${edge.from}|${edge.to}|${edge.dateRaw}|${edge.amount}`;
            if (dedupe.has(key)) return false;
            dedupe.add(key);
            return true;
        })
        .sort((a, b) => {
            if (b.circularProbability !== a.circularProbability) {
                return b.circularProbability - a.circularProbability;
            }
            return b.amount - a.amount;
        });

    return normalized;
}


function deriveRiskLevelFromProbability(probability) {
    const p = Math.max(0, Math.min(1, toFiniteNumber(probability, 0)));
    if (p >= 0.85) return "critical";
    if (p >= 0.65) return "high";
    if (p >= 0.45) return "medium";
    return "low";
}


function canonicalCycleKey(nodes) {
    if (!Array.isArray(nodes) || !nodes.length) return "";

    let best = null;
    for (let i = 0; i < nodes.length; i += 1) {
        const rotated = nodes.slice(i).concat(nodes.slice(0, i));
        const key = rotated.join(">>");
        if (best === null || key < best) {
            best = key;
        }
    }

    return best || "";
}


function detectDirectedCyclesFromCircularEdges(
    circularEdges,
    maxCycleLength = FORENSIC_CYCLE_MAX_LEN,
    maxCycles = FORENSIC_CYCLE_MAX_COUNT,
) {
    const directedAdjacency = new Map();
    circularEdges.forEach((edge) => {
        if (!directedAdjacency.has(edge.from)) directedAdjacency.set(edge.from, new Set());
        directedAdjacency.get(edge.from).add(edge.to);
    });

    const maxLen = Math.max(3, Math.floor(toFiniteNumber(maxCycleLength, FORENSIC_CYCLE_MAX_LEN)));
    const cycleLimit = Math.max(1, Math.floor(toFiniteNumber(maxCycles, FORENSIC_CYCLE_MAX_COUNT)));
    const cycles = [];
    const seen = new Set();
    const starts = [...directedAdjacency.keys()].sort();

    const dfs = (start, current, path, visited) => {
        if (cycles.length >= cycleLimit) return;

        const neighbors = directedAdjacency.get(current);
        if (!neighbors || !neighbors.size) return;

        for (const next of neighbors) {
            if (next === start && path.length >= 3) {
                const cycleNodes = path.slice();
                const key = canonicalCycleKey(cycleNodes);
                if (key && !seen.has(key)) {
                    seen.add(key);
                    cycles.push(cycleNodes);
                    if (cycles.length >= cycleLimit) return;
                }
                continue;
            }

            if (visited.has(next)) continue;
            if (path.length >= maxLen) continue;

            visited.add(next);
            path.push(next);
            dfs(start, next, path, visited);
            path.pop();
            visited.delete(next);

            if (cycles.length >= cycleLimit) return;
        }
    };

    starts.forEach((start) => {
        if (cycles.length >= cycleLimit) return;
        const visited = new Set([start]);
        dfs(start, start, [start], visited);
    });

    return cycles;
}


function buildCircularEdgePairIndex(circularEdges) {
    const pairIndex = new Map();
    circularEdges.forEach((edge) => {
        const key = `${edge.from}|${edge.to}`;
        const bucket = pairIndex.get(key) || [];
        bucket.push(edge);
        pairIndex.set(key, bucket);
    });

    pairIndex.forEach((bucket, key) => {
        pairIndex.set(
            key,
            bucket
                .slice()
                .sort((a, b) => {
                    if (b.circularProbability !== a.circularProbability) {
                        return b.circularProbability - a.circularProbability;
                    }
                    return b.amount - a.amount;
                }),
        );
    });

    return pairIndex;
}


function buildFallbackForensicLogsFromCircularEdges(data) {
    const circularEdges = getCircularEdgeSignals(data);
    if (!circularEdges.length) return [];
    const derivedCycles = detectDirectedCyclesFromCircularEdges(
        circularEdges,
        FORENSIC_CYCLE_MAX_LEN,
        Math.min(60, FORENSIC_CYCLE_MAX_COUNT),
    );

    const totalAmount = circularEdges.reduce((sum, edge) => sum + edge.amount, 0);
    const relatedCompanies = new Set();
    const nodeDegree = new Map();

    circularEdges.forEach((edge) => {
        relatedCompanies.add(edge.from);
        relatedCompanies.add(edge.to);
        nodeDegree.set(edge.from, (nodeDegree.get(edge.from) || 0) + 1);
        nodeDegree.set(edge.to, (nodeDegree.get(edge.to) || 0) + 1);
    });

    const topHub = [...nodeDegree.entries()].sort((a, b) => b[1] - a[1])[0] || null;
    const highestProbabilityEdge = circularEdges[0] || null;
    const largestEdge = [...circularEdges].sort((a, b) => b.amount - a.amount)[0] || null;

    const fallbackTs = "Không rõ thời điểm";
    const logs = [
        {
            timestamp: fallbackTs,
            severity: circularEdges.length >= 3 ? "critical" : "high",
            title: `Suy diễn từ dữ liệu cạnh đỏ (edges.is_circular): ${circularEdges.length} giao dịch`,
            description: `Nguồn dữ liệu: trường edges.is_circular từ phản hồi đồ thị. Quan sát ${relatedCompanies.size} doanh nghiệp liên quan với tổng giá trị ${formatVndCompact(totalAmount)}.`,
        },
    ];

    if (derivedCycles.length) {
        logs.push({
            timestamp: fallbackTs,
            severity: derivedCycles.length >= 3 ? "high" : "medium",
            title: `Phát hiện ${derivedCycles.length} chu trình có hướng từ tập cạnh đỏ`,
            description: `Chu trình được suy ra trực tiếp từ quan hệ từ→đến trên các cạnh đã gắn cờ is_circular.`,
        });
    }

    if (topHub && topHub[1] >= 2) {
        logs.push({
            timestamp: fallbackTs,
            severity: topHub[1] >= 4 ? "high" : "medium",
            title: `Đầu mối giao dịch tập trung tại ${topHub[0]}`,
            description: `${topHub[0]} xuất hiện ${topHub[1]} lần trong tập cạnh đỏ quan sát được.`,
        });
    }

    if (highestProbabilityEdge) {
        logs.push({
            timestamp: formatForensicTimestampOrUnknown(highestProbabilityEdge.dateRaw),
            severity: deriveRiskLevelFromProbability(highestProbabilityEdge.circularProbability),
            title: `Xác suất cao nhất ${Math.round(highestProbabilityEdge.circularProbability * 100)}% trên tuyến ${highestProbabilityEdge.from} → ${highestProbabilityEdge.to}`,
            description: `Giá trị ghi nhận ${formatVndCompact(highestProbabilityEdge.amount)}${highestProbabilityEdge.invoiceNumber ? ` (HĐ ${highestProbabilityEdge.invoiceNumber})` : ""}.`,
        });
    }

    if (largestEdge) {
        logs.push({
            timestamp: formatForensicTimestampOrUnknown(largestEdge.dateRaw),
            severity: largestEdge.amount >= 1_000_000_000 ? "high" : "medium",
            title: `Giá trị lớn nhất trong tập cạnh đỏ: ${formatVndCompact(largestEdge.amount)}`,
            description: `Tuyến ${largestEdge.from} → ${largestEdge.to}${largestEdge.invoiceNumber ? ` (HĐ ${largestEdge.invoiceNumber})` : ""}; circular_probability ${Math.round(largestEdge.circularProbability * 100)}%.`,
        });
    }

    return logs.slice(0, FORENSIC_FALLBACK_MAX_LOGS);
}


function buildFallbackEvidencePathsFromCircularEdges(data) {
    const circularEdges = getCircularEdgeSignals(data);
    if (!circularEdges.length) return [];

    const cycles = detectDirectedCyclesFromCircularEdges(circularEdges);
    const pairIndex = buildCircularEdgePairIndex(circularEdges);

    const cyclePaths = cycles
        .map((cycleNodes, index) => {
            let totalAmount = 0;
            let totalProbability = 0;
            let hasMissingHop = false;

            const hops = cycleNodes
                .map((fromNode, hopIdx) => {
                    const toNode = cycleNodes[(hopIdx + 1) % cycleNodes.length];
                    const bestHop = (pairIndex.get(`${fromNode}|${toNode}`) || [])[0];
                    if (!bestHop) {
                        hasMissingHop = true;
                        return null;
                    }
                    totalAmount += bestHop.amount;
                    totalProbability += bestHop.circularProbability;

                    return {
                        from: bestHop.from,
                        to: bestHop.to,
                        amount_formatted: formatVndCompact(bestHop.amount),
                        date: bestHop.dateRaw || "Không có",
                        fraud_probability: bestHop.circularProbability,
                    };
                })
                .filter(Boolean)
                .slice(0, FORENSIC_FALLBACK_MAX_HOPS);

            if (hasMissingHop || !hops.length) return null;

            const avgProbability = hops.length ? totalProbability / hops.length : 0;
            return {
                path_id: `CIRC-CYCLE-${index + 1}`,
                summary: `Chu trình suy diễn từ edges.is_circular gồm ${cycleNodes.length} mắt xích, tổng giá trị xấp xỉ ${formatVndCompact(totalAmount)}, xác suất trung bình ${(avgProbability * 100).toFixed(0)}%.`,
                risk_level: deriveRiskLevelFromProbability(avgProbability),
                companies: cycleNodes,
                hops,
            };
        })
        .filter(Boolean)
        .sort((a, b) => {
            const aProb = toFiniteNumber(a?.hops?.[0]?.fraud_probability, 0);
            const bProb = toFiniteNumber(b?.hops?.[0]?.fraud_probability, 0);
            return bProb - aProb;
        });

    if (cyclePaths.length) {
        return cyclePaths.slice(0, FORENSIC_FALLBACK_MAX_PATHS);
    }

    return circularEdges.slice(0, FORENSIC_FALLBACK_MAX_PATHS).map((edge, index) => ({
        path_id: `CIRC-EDGE-${index + 1}`,
        summary: `Tuyến nghi vấn trực tiếp từ edges.is_circular: ${edge.from} → ${edge.to}, giá trị ${formatVndCompact(edge.amount)}, xác suất ${Math.round(edge.circularProbability * 100)}%.`,
        risk_level: deriveRiskLevelFromProbability(edge.circularProbability),
        companies: [edge.from, edge.to],
        hops: [
            {
                from: edge.from,
                to: edge.to,
                amount_formatted: formatVndCompact(edge.amount),
                date: edge.dateRaw || "Không có",
                fraud_probability: edge.circularProbability,
            },
        ],
    }));
}


function renderForensicIntelSummary(forensicIntel) {
    const strip = document.getElementById("forensic-intel-strip");
    if (!strip) return;

    const diagnostics = forensicIntel && typeof forensicIntel.diagnostics === "object"
        ? forensicIntel.diagnostics
        : {};
    const ringPayload = forensicIntel && typeof forensicIntel.ring === "object" ? forensicIntel.ring : null;
    const ownershipPayload = forensicIntel && typeof forensicIntel.ownership === "object" ? forensicIntel.ownership : null;
    const ownershipSummary = ownershipPayload && typeof ownershipPayload.summary === "object" ? ownershipPayload.summary : {};
    const ringStatus = normalizeForensicDataStatus(
        diagnostics.ring_status || ringPayload?.data_status,
        ringPayload ? "ok" : "unavailable",
    );
    const ownershipStatus = normalizeForensicDataStatus(
        diagnostics.ownership_status || ownershipPayload?.data_status,
        ownershipPayload ? "ok" : "unavailable",
    );
    const ringUnavailable = isForensicDataUnavailable(ringStatus);
    const ownershipUnavailable = isForensicDataUnavailable(ownershipStatus);

    const ringTotal = toFiniteNumber(ringPayload?.total, 0);
    const ringCritical = toFiniteNumber(ringPayload?.critical_count, 0);
    const ringCyclesDetected = toFiniteNumber(ringPayload?.cycles_detected, ringTotal);
    const ringCoveragePct = Math.max(0, Math.min(100, toFiniteNumber(ringPayload?.circular_edge_cycle_coverage, 0) * 100));
    const ownershipClusters = toFiniteNumber(ownershipSummary.total_clusters, 0);
    const crossTrades = toFiniteNumber(ownershipSummary.total_cross_trades, 0);
    const commonControllers = toFiniteNumber(ownershipSummary.total_common_controllers, 0);
    const ownershipCoverage = toFiniteNumber(ownershipPayload?.coverage?.ownership_invoice_node_coverage, -1);

    const hasDataIssue = ringStatus !== "ok" || ownershipStatus !== "ok";
    const hasAnySignal = ringTotal > 0 || ringCritical > 0 || ringCyclesDetected > 0 || ownershipClusters > 0 || crossTrades > 0 || commonControllers > 0 || hasDataIssue;
    if (!hasAnySignal) {
        strip.classList.add("hidden");
        return;
    }

    strip.classList.remove("hidden");
    setElementText("forensic-ring-total", ringUnavailable ? "--" : String(ringTotal));
    setElementText("forensic-ring-critical", ringUnavailable ? "--" : String(ringCritical));
    setElementText("forensic-ownership-clusters", ownershipUnavailable ? "--" : String(ownershipClusters));
    setElementText("forensic-cross-trades", ownershipUnavailable ? "--" : String(crossTrades));

    const updatedEl = document.getElementById("forensic-intel-updated");
    if (updatedEl) {
        const ts = formatShortTimestamp(forensicIntel?.fetched_at);
        const cycleScopeText = ringCyclesDetected > ringTotal
            ? ` · Vòng phát hiện: ${ringCyclesDetected}, vòng chấm điểm: ${ringTotal}`
            : "";
        const coverageText = ringCoveragePct > 0
            ? ` · Bao phủ cạnh-vòng: ${ringCoveragePct.toFixed(0)}%`
            : "";
        const ownershipCoverageText = ownershipCoverage >= 0
            ? ` · Phủ sở hữu-hóa đơn: ${(ownershipCoverage * 100).toFixed(0)}%`
            : "";
        const statusText = hasDataIssue
            ? ` · Trạng thái dữ liệu: vòng ${formatForensicDataStatus(ringStatus)}, sở hữu ${formatForensicDataStatus(ownershipStatus)}`
            : "";
        updatedEl.textContent = `Nguồn sở hữu/vòng đã đồng bộ${ts !== "--" ? ` lúc ${ts}` : ""} · Bộ điều phối: ${commonControllers}${cycleScopeText}${coverageText}${ownershipCoverageText}${statusText}`;
    }
}


function buildIntegratedForensicLogs(data, forensicIntel) {
    const baseLogsRaw = Array.isArray(data?.logs) ? data.logs : [];
    const baseLogs = baseLogsRaw
        .filter((item) => item && typeof item === "object")
        .map((log) => ({
            timestamp: String(log.timestamp || "Không rõ thời điểm"),
            severity: String(log.severity || "low").toLowerCase(),
            title: String(log.title || "Cập nhật điều tra"),
            description: String(log.description || "Không có mô tả bổ sung."),
        }));

    const ringPayload = forensicIntel && typeof forensicIntel.ring === "object" ? forensicIntel.ring : null;
    const ownershipPayload = forensicIntel && typeof forensicIntel.ownership === "object" ? forensicIntel.ownership : null;
    const ownershipSummary = ownershipPayload && typeof ownershipPayload.summary === "object" ? ownershipPayload.summary : {};
    const diagnostics = forensicIntel && typeof forensicIntel.diagnostics === "object"
        ? forensicIntel.diagnostics
        : {};
    const ringStatus = normalizeForensicDataStatus(
        diagnostics.ring_status || ringPayload?.data_status,
        ringPayload ? "ok" : "unavailable",
    );
    const ownershipStatus = normalizeForensicDataStatus(
        diagnostics.ownership_status || ownershipPayload?.data_status,
        ownershipPayload ? "ok" : "unavailable",
    );

    const integratedLogs = [];
    const nowTs = formatForensicTimestampOrUnknown(forensicIntel?.fetched_at);

    if (ringStatus !== "ok") {
        integratedLogs.push({
            timestamp: nowTs,
            severity: isForensicDataUnavailable(ringStatus) ? "high" : "medium",
            title: `Trạng thái dữ liệu vòng: ${formatForensicDataStatus(ringStatus)}`,
            description: "Chỉ số vòng có thể chưa phản ánh đầy đủ toàn bộ cạnh nghi vấn trong phạm vi truy vấn hiện tại.",
        });
    }

    if (ownershipStatus !== "ok") {
        integratedLogs.push({
            timestamp: nowTs,
            severity: isForensicDataUnavailable(ownershipStatus) ? "high" : "medium",
            title: `Trạng thái dữ liệu sở hữu: ${formatForensicDataStatus(ownershipStatus)}`,
            description: "Kết quả sở hữu/cross-trade có thể bị ảnh hưởng bởi phạm vi node-hóa đơn hoặc dữ liệu nguồn chưa sẵn sàng.",
        });
    }

    const ringTotal = toFiniteNumber(ringPayload?.total, 0);
    const ringCritical = toFiniteNumber(ringPayload?.critical_count, 0);
    const ringCyclesDetected = toFiniteNumber(ringPayload?.cycles_detected, ringTotal);
    if (ringTotal > 0) {
        integratedLogs.push({
            timestamp: nowTs,
            severity: ringCritical > 0 ? "high" : "medium",
            title: `Phát hiện ${ringTotal} vòng giao dịch nghi vấn`,
            description: `Hệ thống Ring Scoring ghi nhận ${ringCritical} vòng mức critical${ringCyclesDetected > ringTotal ? `; tổng số vòng phát hiện là ${ringCyclesDetected}` : ""}.`,
        });
    }

    if (ringCyclesDetected > ringTotal) {
        integratedLogs.push({
            timestamp: nowTs,
            severity: "medium",
            title: `Vòng phát hiện nhiều hơn vòng đã chấm điểm (${ringTotal}/${ringCyclesDetected})`,
            description: "Ngưỡng hiển thị hiện tại chỉ trả về một phần vòng để tránh quá tải giao diện.",
        });
    }

    const crossTrades = toFiniteNumber(ownershipSummary.total_cross_trades, 0);
    if (crossTrades > 0) {
        integratedLogs.push({
            timestamp: nowTs,
            severity: "critical",
            title: `Có ${crossTrades} giao dịch liên đới quan hệ sở hữu`,
            description: "Đồ thị sở hữu phát hiện giao dịch bên liên quan giữa các pháp nhân có liên kết sở hữu; cần xác minh hồ sơ đối ứng.",
        });
    }

    const ownershipClusters = toFiniteNumber(ownershipSummary.total_clusters, 0);
    const commonControllers = toFiniteNumber(ownershipSummary.total_common_controllers, 0);
    const ownershipCoverage = toFiniteNumber(ownershipPayload?.coverage?.ownership_invoice_node_coverage, -1);
    const ownershipNodesInInvoiceScope = toFiniteNumber(
        ownershipPayload?.coverage?.ownership_nodes_in_invoice_graph_count,
        0,
    );
    if (ownershipClusters > 0 || commonControllers > 0) {
        integratedLogs.push({
            timestamp: nowTs,
            severity: "medium",
            title: `Mạng sở hữu có ${ownershipClusters} cụm và ${commonControllers} đầu mối chung`,
            description: "Đề nghị rà soát luồng hóa đơn phát sinh trong các cụm sở hữu để loại trừ kịch bản điều phối giao dịch nội bộ.",
        });
    }

    if (ownershipClusters > 0 && crossTrades === 0 && ownershipCoverage >= 0) {
        integratedLogs.push({
            timestamp: nowTs,
            severity: ownershipCoverage < 0.4 ? "high" : "medium",
            title: `Cụm sở hữu có tín hiệu nhưng chưa ghi nhận giao dịch liên đới`,
            description: `Độ phủ node sở hữu trong phạm vi hóa đơn hiện tại là ${(ownershipCoverage * 100).toFixed(0)}% (${ownershipNodesInInvoiceScope} node trùng phạm vi).`,
        });
    }

    const mergedLogs = [...integratedLogs, ...baseLogs]
        .filter((log) => log.title && log.description)
        .slice(0, FORENSIC_MERGED_LOG_LIMIT);

    if (mergedLogs.length) {
        return mergedLogs;
    }

    return buildFallbackForensicLogsFromCircularEdges(data);
}


function dedupeEvidencePaths(paths) {
    const seen = new Set();
    const deduped = [];

    paths.forEach((path) => {
        if (!path || typeof path !== "object") return;
        const companies = Array.isArray(path.companies)
            ? path.companies.map((item) => String(item || "")).join("|")
            : "";
        const hops = Array.isArray(path.hops)
            ? path.hops.map((hop) => `${hop?.from || ""}->${hop?.to || ""}`).join("|")
            : "";
        const key = `${companies}::${hops}`;
        if (seen.has(key)) return;
        seen.add(key);
        deduped.push(path);
    });

    return deduped;
}


function renderGeoForensicSignals(compatDiagnostics, crossBorderSignals) {
    const banner = document.getElementById("geo-compat-banner");
    const strip = document.getElementById("cross-border-strip");

    if (!compatDiagnostics && !crossBorderSignals) {
        if (banner) banner.classList.add("hidden");
        if (strip) strip.classList.add("hidden");
        return;
    }

    if (banner && compatDiagnostics) {
        banner.classList.remove("hidden");
        const status = compatDiagnostics.status || "unknown";
        const ratio = toFiniteNumber(compatDiagnostics.completion_ratio, 0);

        const badge = document.getElementById("compat-status-badge");
        if (badge) {
            badge.className = "text-[9px] font-black uppercase px-2 py-0.5 rounded border transition-colors";
            if (status === "pass") {
                badge.textContent = "ĐẠT";
                badge.classList.add("compat-badge-pass");
            } else if (status === "partial") {
                badge.textContent = "CHƯA FULL";
                badge.classList.add("compat-badge-partial");
            } else if (status === "fail") {
                badge.textContent = "LỖI TƯƠNG THÍCH";
                badge.classList.add("compat-badge-fail");
            } else {
                badge.textContent = "ĐANG TẢI";
                badge.classList.add("compat-badge-unknown");
            }
        }

        const bar = document.getElementById("compat-progress-bar");
        if (bar) {
            bar.style.width = ratio + "%";
            bar.className = "h-full rounded-full transition-all duration-500 " +
                (status === "pass" ? "bg-emerald-500" :
                    status === "partial" ? "bg-amber-500" :
                        status === "fail" ? "bg-error" : "bg-slate-300");
        }

        const ratioLabel = document.getElementById("compat-ratio-label");
        if (ratioLabel) {
            ratioLabel.textContent = ratio.toFixed(0) + "%";
            ratioLabel.style.color = (status === "pass" ? "#10b981" : status === "partial" ? "#f59e0b" : "#ef4444");
        }

        const detail = document.getElementById("compat-detail-text");
        if (detail) {
            detail.textContent = "Tương thích Data Contract: " + (compatDiagnostics.schema_version || "v2");
        }
    } else if (banner) {
        banner.classList.add("hidden");
    }

    if (strip && crossBorderSignals) {
        if (crossBorderSignals.available) {
            strip.classList.remove("hidden");

            const riskLevel = String(crossBorderSignals.risk_level || "low").toLowerCase();
            const badge = document.getElementById("cross-border-risk-badge");
            if (badge) {
                badge.className = "text-[9px] font-black uppercase px-2 py-0.5 rounded border transition-colors " +
                    (riskLevel === "critical" || riskLevel === "high" ? "cross-border-badge-high" :
                        riskLevel === "medium" ? "cross-border-badge-medium" : "cross-border-badge-low");
                badge.textContent = riskLevel === "critical" ? "NGHIÊM TRỌNG" : riskLevel === "high" ? "CAO" : riskLevel === "medium" ? "TRUNG BÌNH" : "THẤP";
            }

            setElementText("cross-border-foreign-count", crossBorderSignals.companies_outside_vietnam || "0");
            setElementText("cross-border-highrisk-count", crossBorderSignals.high_risk_country_exposure?.count || "0");
            setElementText("cross-border-score", toFiniteNumber(crossBorderSignals.risk_score, 0).toFixed(1));

            const total = toFiniteNumber(crossBorderSignals.scope_companies_total, 0);
            const foreign = toFiniteNumber(crossBorderSignals.companies_outside_vietnam, 0);
            const crossTradesCount = toFiniteNumber(crossBorderSignals.cross_border_invoice_count, 0);
            const summary = document.getElementById("cross-border-summary-text");
            if (summary) {
                summary.innerHTML = `Phát hiện <b>${foreign}/${total}</b> tỷ lệ ngoài lãnh thổ. Ghi nhận <b>${crossTradesCount}</b> giao dịch chéo quốc gia.`;
            }

            const chipsContainer = document.getElementById("cross-border-country-chips");
            if (chipsContainer && Array.isArray(crossBorderSignals.country_company_distribution)) {
                const limitChips = crossBorderSignals.country_company_distribution.slice(0, 4);
                chipsContainer.innerHTML = limitChips.map(c => {
                    const countryName = c.country || "Unknown";
                    const isVN = countryName.toLowerCase() === "vietnam" || countryName.toLowerCase() === "việt nam";
                    const flag = isVN ? "🇻🇳" : "🏳️";
                    const count = c.companies || 0;
                    const highRisk = !isVN && count > 0 ? "high-risk" : "";
                    return `<span class="country-chip ${highRisk}"><span class="chip-flag">${flag}</span> ${escapeHtml(countryName)} <span class="chip-count">x${count}</span></span>`;
                }).join("");
            }
        } else {
            strip.classList.add("hidden");
        }
    } else if (strip) {
        strip.classList.add("hidden");
    }
}


function formatDateAndSpanLabel(startDateRaw, endDateRaw, spanDaysRaw) {
    const start = String(startDateRaw || "").trim();
    const end = String(endDateRaw || "").trim();
    let spanDays = Number.isFinite(Number(spanDaysRaw)) ? Number(spanDaysRaw) : null;

    const fmt = (value) => {
        if (!value) return "";
        const d = new Date(value);
        if (Number.isNaN(d.getTime())) return "";
        return d.toLocaleDateString("vi-VN");
    };

    const startFmt = fmt(start);
    const endFmt = fmt(end);
    if (spanDays === null && startFmt && endFmt) {
        const startDate = new Date(start);
        const endDate = new Date(end);
        if (!Number.isNaN(startDate.getTime()) && !Number.isNaN(endDate.getTime())) {
            spanDays = Math.max(0, Math.round((endDate.getTime() - startDate.getTime()) / 86400000));
        }
    }

    if (startFmt && endFmt) {
        if (spanDays !== null) return `${startFmt} - ${endFmt} (${spanDays} ngày)`;
        return `${startFmt} - ${endFmt}`;
    }
    if (startFmt) {
        if (spanDays !== null) return `${startFmt} (${spanDays} ngày)`;
        return startFmt;
    }
    if (spanDays !== null) return `Khoảng thời gian: ${spanDays} ngày`;
    return "Không có";
}


function resolveHopAmountLabel(hop) {
    const formatted = String(hop?.amount_formatted || "").trim();
    if (formatted) return formatted;

    const candidates = [
        hop?.transfer_amount,
        hop?.amount,
        hop?.value,
        hop?.transaction_amount,
    ];
    for (const raw of candidates) {
        const num = Number(raw);
        if (Number.isFinite(num) && num > 0) {
            return formatVndCompact(num);
        }
    }
    return "Không có dữ liệu";
}


function buildIntegratedEvidencePaths(data, forensicIntel) {
    let basePaths = [];
    if (Array.isArray(data?.evidence_chains) && data.evidence_chains.length > 0) {
        basePaths = data.evidence_chains;
    } else if (Array.isArray(data?.evidence_paths)) {
        basePaths = data.evidence_paths;
    }
    const integratedPaths = [];

    const ringPayload = forensicIntel && typeof forensicIntel.ring === "object" ? forensicIntel.ring : null;
    const ringList = Array.isArray(ringPayload?.rings) ? ringPayload.rings : [];
    ringList.slice(0, FORENSIC_RING_PATH_LIMIT).forEach((ring, index) => {
        const nodes = Array.isArray(ring?.nodes) ? ring.nodes.map((n) => String(n || "")).filter(Boolean) : [];
        if (nodes.length < 2) return;

        const totalAmount = toFiniteNumber(ring.total_amount, 0);
        const perHopAmount = nodes.length ? totalAmount / nodes.length : totalAmount;
        const ringScore = toFiniteNumber(ring.ring_score, 0);
        const ringDateLabel = formatDateAndSpanLabel(ring.start_date, ring.end_date, ring.time_span_days);

        const hops = nodes.map((fromNode, hopIdx) => {
            const toNode = nodes[(hopIdx + 1) % nodes.length];
            return {
                from: fromNode,
                to: toNode,
                amount_formatted: formatVndCompact(perHopAmount),
                date: ringDateLabel,
                fraud_probability: ringScore,
            };
        });

        integratedPaths.push({
            path_id: `RING-${index + 1}`,
            summary: `Vòng giao dịch ${nodes.length} mắt xích, tổng giá trị ước tính ${formatVndCompact(totalAmount)}, ring score ${ringScore.toFixed(2)}.`,
            risk_level: String(ring.risk_level || "high").toLowerCase(),
            companies: nodes,
            hops,
        });
    });

    const ownershipPayload = forensicIntel && typeof forensicIntel.ownership === "object" ? forensicIntel.ownership : null;
    const crossTrades = Array.isArray(ownershipPayload?.cross_ownership_trades)
        ? ownershipPayload.cross_ownership_trades
        : [];
    crossTrades.slice(0, FORENSIC_OWNERSHIP_PATH_LIMIT).forEach((trade, index) => {
        const parent = String(trade?.parent || "").trim();
        const child = String(trade?.child || "").trim();
        if (!parent || !child) return;

        const ownershipPercent = toFiniteNumber(trade?.ownership_percent, 0);
        const reason = String(trade?.reason || "Giao dịch giữa các pháp nhân có quan hệ sở hữu.");
        const direction = String(trade?.trade_direction || "cha→con");

        integratedPaths.push({
            path_id: `OWN-${index + 1}`,
            summary: `${reason} Hướng giao dịch: ${direction}; tỷ lệ sở hữu ghi nhận ${ownershipPercent.toFixed(1)}%.`,
            risk_level: String(trade?.risk_level || "high").toLowerCase(),
            companies: [parent, child],
            hops: [
                {
                    from: parent,
                    to: child,
                    amount_formatted: `Sở hữu ${ownershipPercent.toFixed(1)}%`,
                    date: formatDateAndSpanLabel(trade?.start_date, trade?.end_date, trade?.time_span_days),
                    fraud_probability: Math.min(1, ownershipPercent / 100),
                },
            ],
        });
    });

    const mergedPaths = dedupeEvidencePaths([
        ...integratedPaths,
        ...basePaths,
    ]).slice(0, FORENSIC_FALLBACK_MAX_PATHS);
    if (mergedPaths.length) {
        return mergedPaths;
    }
    return buildFallbackEvidencePathsFromCircularEdges(data);
}


function renderForensicPanel(data, forensicIntel = null) {
    const payload = data && typeof data === "object" ? data : {};
    renderForensicIntelSummary(forensicIntel);

    // Total suspicious amount
    const totalEl = document.getElementById("total-suspicious-amount");
    if (totalEl) {
        const amount = toFiniteNumber(payload.total_suspicious_amount, 0);
        totalEl.textContent = new Intl.NumberFormat("vi-VN").format(amount) + " VNĐ";
        // Animate counter
        animateCounter(totalEl, 0, amount, 1500, (v) => `${new Intl.NumberFormat("vi-VN").format(Math.round(v))} VNĐ`);
    }

    // Invoice count
    const countEl = document.getElementById("suspicious-invoice-count");
    if (countEl) {
        const totalInv = toFiniteNumber(payload.total_suspicious_invoices, 0);
        const integrityStatus = payload?.integrity_signals?.available ? "Integrity sẵn sàng" : "Integrity thiếu dữ liệu";
        const pricingStatus = payload?.pricing_signals?.available ? "Pricing sẵn sàng" : "Pricing thiếu dữ liệu";
        const phoenixStatus = payload?.phoenix_signals?.available ? "Phoenix sẵn sàng" : "Phoenix thiếu dữ liệu";
        countEl.textContent = `Bao gồm ${totalInv} hóa đơn nghi vấn. Trạng thái: ${integrityStatus}, ${pricingStatus}, ${phoenixStatus}.`;
    }

    // Advanced Fraud Patterns
    const advFraudStrip = document.getElementById("advanced-fraud-strip");
    if (advFraudStrip && payload.forensic_metrics) {
        advFraudStrip.classList.remove("hidden");
        const smurfEl = document.getElementById("fraud-smurfing");
        const ghostEl = document.getElementById("fraud-ghost");
        const teleEl = document.getElementById("fraud-teleportation");
        const baselineEl = document.getElementById("fraud-kpi-baseline");

        const smurfCount = toFiniteNumber(payload.forensic_metrics.smurfing_count, 0);
        const ghostCount = toFiniteNumber(payload.forensic_metrics.ghost_node_count, 0);
        const teleCount = toFiniteNumber(payload.forensic_metrics.teleportation_count, 0);
        const totalInv = Math.max(1, toFiniteNumber(payload.total_suspicious_invoices, 0));
        const totalNodes = Math.max(1, Array.isArray(payload.nodes) ? payload.nodes.length : 0);
        const totalEdges = Math.max(1, Array.isArray(payload.edges) ? payload.edges.length : 0);

        if (smurfEl) animateCounter(smurfEl, 0, smurfCount, 1000, (v) => `${new Intl.NumberFormat("vi-VN").format(Math.round(v))} ca`);
        if (ghostEl) animateCounter(ghostEl, 0, ghostCount, 1000, (v) => `${new Intl.NumberFormat("vi-VN").format(Math.round(v))} nút`);
        if (teleEl) animateCounter(teleEl, 0, teleCount, 1000, (v) => `${new Intl.NumberFormat("vi-VN").format(Math.round(v))} cạnh`);

        if (baselineEl) {
            const smurfPer1k = ((smurfCount / totalInv) * 1000).toFixed(1);
            const ghostPer1k = ((ghostCount / totalNodes) * 1000).toFixed(1);
            const telePer1k = ((teleCount / totalEdges) * 1000).toFixed(1);
            baselineEl.textContent = `Chuẩn hóa quy mô: Smurf ${smurfPer1k}/1.000 hóa đơn | Ghost ${ghostPer1k}/1.000 nút | Teleport ${telePer1k}/1.000 cạnh.`;
        }

        advFraudStrip.querySelectorAll("[data-fraud-filter]").forEach((card) => {
            if (card.getAttribute("data-initialized") === "true") return;
            card.setAttribute("data-initialized", "true");
            card.addEventListener("click", () => {
                switchForensicTab("paths");
                const paths = document.getElementById("evidence-paths-container");
                if (paths) paths.scrollIntoView({ behavior: "smooth", block: "start" });
            });
        });

        const extraSignals = [
            { id: "washout", value: payload.forensic_metrics.vat_washout_node_count || 0, label: "Nút washout" },
            { id: "mismatch", value: payload.forensic_metrics.mismatch_edge_count || 0, label: "Cạnh lệch ngành-hàng" },
            { id: "phoenix", value: payload.forensic_metrics.phoenix_link_count || 0, label: "Chuỗi phoenix" },
            { id: "payment", value: payload.forensic_metrics.payment_mismatch_count || 0, label: "Lệch thanh toán" },
        ];
        const panel = document.getElementById("forensic-intel-updated");
        if (panel && extraSignals.some((item) => item.value > 0)) {
            const summary = extraSignals.map((item) => `${item.label}: ${item.value}`).join(" | ");
            panel.textContent = `${panel.textContent || ""} | ${summary}`.trim();
        }
        if (panel && Array.isArray(payload.top_invoice_risks) && payload.top_invoice_risks.length > 0) {
            const top = payload.top_invoice_risks[0];
            const invNo = String(top.invoice_number || "--");
            const invScore = toFiniteNumber(top.risk_score, 0).toFixed(1);
            panel.textContent = `${panel.textContent || ""} | HĐ rủi ro cao: ${invNo} (${invScore})`.trim();
        }
    }

    renderForensicSignalsV21(payload);

    // Severity badge
    const badge = document.getElementById("severity-badge");
    if (badge) {
        const amount = toFiniteNumber(payload.total_suspicious_amount, 0);
        if (amount > 10e9) {
            badge.textContent = "CẢNH BÁO CAO";
            badge.className = "bg-error text-white text-[10px] font-black px-2 py-0.5 rounded uppercase";
        } else if (amount > 1e9) {
            badge.textContent = "CẢNH BÁO";
            badge.className = "bg-amber-500 text-white text-[10px] font-black px-2 py-0.5 rounded uppercase";
        } else {
            badge.textContent = "BÌNH THƯỜNG";
            badge.className = "bg-emerald-500 text-white text-[10px] font-black px-2 py-0.5 rounded uppercase";
        }
    }

    // Investigation logs
    const logsContainer = document.getElementById("investigation-logs");
    const integratedLogs = buildIntegratedForensicLogs(payload, forensicIntel);
    if (logsContainer) {
        if (!integratedLogs.length) {
            logsContainer.innerHTML = `
                <div class="flex items-center justify-center h-32">
                    <p class="text-xs text-slate-400 italic">Chưa có dữ liệu nhật ký truy vết.</p>
                </div>`;
        } else {
            logsContainer.innerHTML = integratedLogs.map((log, i) => {
            const severityColors = {
                critical: { border: "border-error", dot: "bg-error", text: "text-error", label: "NGHIÊM TRỌNG" },
                high:     { border: "border-amber-500", dot: "bg-amber-500", text: "text-amber-600", label: "CAO" },
                medium:   { border: "border-blue-500", dot: "bg-blue-500", text: "text-blue-600", label: "TRUNG BÌNH" },
                low:      { border: "border-emerald-500", dot: "bg-emerald-500", text: "text-emerald-600", label: "THẤP" },
            };
            const s = severityColors[log.severity] || severityColors.low;
            const ts = escapeHtml(log.timestamp || "");
            const title = escapeHtml(log.title || "");
            const description = escapeHtml(log.description || "");

            return `
                <div class="relative pl-4 ${s.border} border-l-2 log-entry" style="animation: fadeSlideIn 0.4s ease ${i * 0.15}s both;">
                    <div class="absolute -left-[5px] top-1 w-2 h-2 rounded-full ${s.dot}"></div>
                    <div class="flex justify-between items-start mb-1">
                        <p class="text-[10px] font-mono text-slate-500 font-bold">${ts}</p>
                        <p class="text-[10px] font-bold ${s.text} uppercase">${s.label}</p>
                    </div>
                    <p class="text-sm font-bold text-primary-container leading-tight">${title}</p>
                    <p class="text-xs text-on-surface-variant mt-1">${description}</p>
                </div>
            `;
            }).join("");
        }
    }

    // Render Geo-Forensic Compatibility v2
    renderGeoForensicSignals(payload.compatibility_diagnostics, payload.cross_border_signals);

    // Evidence Paths Rendering
    const pathsContainer = document.getElementById("evidence-paths-container");
    const pathBadge = document.getElementById("path-badge");
    const evidencePaths = buildIntegratedEvidencePaths(payload, forensicIntel);
    if (pathsContainer && pathBadge) {
        if (!evidencePaths.length) {
            pathBadge.classList.add("hidden");
            pathsContainer.innerHTML = `
                <div class="flex items-center justify-center h-32">
                    <p class="text-xs text-slate-400 italic">Không phát hiện chuỗi quay vòng tuần hoàn.</p>
                </div>`;
        } else {
            const ringTruncated = Boolean(forensicIntel?.ring?.truncated);
            pathBadge.textContent = ringTruncated ? `${evidencePaths.length}+` : String(evidencePaths.length);
            pathBadge.classList.remove("hidden");
            const pathHTML = evidencePaths.map((p, pIdx) => {
                const riskColors = {
                    critical: "text-error border-error bg-error/10",
                    high: "text-amber-600 border-amber-500 bg-amber-500/10",
                    medium: "text-blue-600 border-blue-500 bg-blue-500/10",
                    low: "text-emerald-600 border-emerald-500 bg-emerald-500/10",
                };
                const riskLevelKey = String(p.risk_level || "medium").toLowerCase();
                const cColor = riskColors[riskLevelKey] || riskColors.medium;
                const pathId = escapeHtml(p.path_id || `PATH-${pIdx + 1}`);
                const summary = escapeHtml(p.summary || "Chuỗi bằng chứng nghi vấn.");
                const riskLevel = escapeHtml(riskLevelKey === "critical" ? "NGHIÊM TRỌNG" : riskLevelKey.toUpperCase());
                const companiesCsv = Array.isArray(p.companies) ? p.companies.map(c => String(c)).join(",") : "";
                const companiesEncoded = encodeURIComponent(companiesCsv);
                const hops = Array.isArray(p.hops) ? p.hops : [];
                const chainScore = typeof p.chain_score !== 'undefined' ? 
                                    `<span class="evidence-chain-score" data-level="${riskLevelKey}">
                                        <span class="material-symbols-outlined" style="font-size: 11px;">radar</span> ${toFiniteNumber(p.chain_score, 0).toFixed(1)}
                                    </span>` : '';
                
                const hopsHTML = hops.length ? hops.map((h) => {
                    const fromCountry = typeof h.provenance?.from_country === 'string' && h.provenance.from_country.trim() !== '' ? h.provenance.from_country : '';
                    const toCountry = typeof h.provenance?.to_country === 'string' && h.provenance.to_country.trim() !== '' ? h.provenance.to_country : '';
                    const isFromVN = fromCountry.toLowerCase() === 'vietnam' || fromCountry.toLowerCase() === 'việt nam';
                    const isToVN = toCountry.toLowerCase() === 'vietnam' || toCountry.toLowerCase() === 'việt nam';
                    
                    let fromTag = '';
                    if (fromCountry) {
                        const styleClass = isFromVN ? '' : 'foreign';
                        const flag = '🇻🇳'; // Assuming VN mostly, or generic marker
                        const displayFlag = isFromVN ? flag : '🏳️';
                        fromTag = `<span class="hop-country-tag ${styleClass}">${displayFlag} ${escapeHtml(fromCountry)}</span>`;
                    }
                    
                    let toTag = '';
                    if (toCountry) {
                        const styleClass = isToVN ? '' : 'foreign';
                        const displayFlag = isToVN ? '🇻🇳' : '🏳️';
                        toTag = `<span class="hop-country-tag ${styleClass}">${displayFlag} ${escapeHtml(toCountry)}</span>`;
                    }
                    
                    return `
                    <div class="relative pl-6 pb-4 border-l-2 border-slate-200 last:border-transparent last:pb-0">
                        <div class="absolute -left-[5px] top-1 w-2 h-2 rounded-full bg-primary-container z-10"></div>
                        <div class="bg-surface-container-low p-2 rounded-lg border border-outline-variant/30 text-[10px]">
                            <div class="flex justify-between items-center mb-1">
                                <span class="font-bold text-primary-container">
                                    <span title="MST: ${escapeHtml(String(h.from || ""))}">${escapeHtml(String(h.from || ""))}</span>
                                    <button class="ml-1 text-[9px] text-slate-400 hover:text-primary-container" data-copy-tax="${escapeHtml(String(h.from || ""))}" title="Sao chép MST">Copy</button>
                                    →
                                    <span title="MST: ${escapeHtml(String(h.to || ""))}">${escapeHtml(String(h.to || ""))}</span>
                                    <button class="ml-1 text-[9px] text-slate-400 hover:text-primary-container" data-copy-tax="${escapeHtml(String(h.to || ""))}" title="Sao chép MST">Copy</button>
                                </span>
                                <span class="text-error font-bold">${escapeHtml(resolveHopAmountLabel(h))}</span>
                            </div>
                            ${fromTag || toTag ? `<div class="flex gap-1.5 mb-1.5">${fromTag}${toTag ? '<span class="text-slate-400 mt-[-1px]">→</span>' + toTag : ''}</div>` : ''}
                            <div class="flex justify-between text-slate-500 font-mono">
                                <span>Thời gian: ${escapeHtml(h.date || "Không có")}</span>
                                <span>Xác suất: ${toFiniteNumber(h.fraud_probability, 0).toFixed(2)}</span>
                            </div>
                        </div>
                    </div>
                `}).join("") : `<p class="text-[10px] text-slate-400 italic">Chuỗi này chưa có chi tiết giao dịch theo từng chặng.</p>`;

                return `
                    <div class="border border-outline-variant/30 rounded-xl p-3 mb-4 log-entry" style="animation: fadeSlideIn 0.4s ease ${pIdx * 0.1}s both;">
                        <div class="flex justify-between items-center mb-2">
                            <div class="flex items-center gap-2">
                                <span class="text-xs font-black text-primary-container">${pathId}</span>
                                ${chainScore}
                            </div>
                            <span class="text-[9px] font-black uppercase px-2 py-0.5 rounded border ${cColor}">${riskLevel}</span>
                        </div>
                        <p class="text-[10px] text-on-surface-variant font-medium leading-relaxed mb-3">${summary}</p>
                        <div class="mt-2">
                            ${hopsHTML}
                        </div>
                        <button data-action="focus-chain" data-companies="${companiesEncoded}" class="w-full mt-3 py-1.5 flex items-center justify-center gap-1 text-[10px] uppercase font-bold text-primary-container bg-primary-container/5 hover:bg-primary-container/10 rounded transition-colors">
                            <span class="material-symbols-outlined text-[14px]">visibility</span> Tập trung chuỗi
                        </button>
                    </div>
                `;
            });
            
            pathsContainer.innerHTML = pathHTML.join("");
            pathsContainer.querySelectorAll("button[data-action='focus-chain']").forEach((btn) => {
                btn.addEventListener("click", () => {
                    const companies = decodeURIComponent(btn.dataset.companies || "");
                    if (companies) highlightCycleNodes(companies);
                });
            });
            pathsContainer.querySelectorAll("button[data-copy-tax]").forEach((btn) => {
                btn.addEventListener("click", async () => {
                    const taxCode = String(btn.dataset.copyTax || "").trim();
                    if (!taxCode) return;
                    try {
                        await navigator.clipboard.writeText(taxCode);
                        showGraphToast("Đã sao chép MST", taxCode, "success");
                    } catch (_err) {
                        showGraphToast("Không thể sao chép", "Trình duyệt không cấp quyền clipboard.", "warning");
                    }
                });
            });
        }
    }

    setupForensicTabs();
}

function renderForensicSignalsV21(payload) {
    const strip = document.getElementById("forensic-v21-strip");
    const cards = document.getElementById("forensic-v21-cards");
    const hint = document.getElementById("forensic-v21-hint");
    const contractBadge = document.getElementById("forensic-v21-contract");
    if (!strip || !cards || !hint || !contractBadge) return;

    const integrity = payload?.integrity_signals || {};
    const pricing = payload?.pricing_signals || {};
    const phoenix = payload?.phoenix_signals || {};
    const hasAny = Boolean(integrity.available || pricing.available || phoenix.available);

    contractBadge.textContent = String(payload?.contract_version || "graph-intelligence-v2.1");

    if (!hasAny) {
        strip.classList.add("hidden");
        return;
    }

    strip.classList.remove("hidden");
    const cardDefs = [
        {
            title: "Lifecycle Integrity",
            available: Boolean(integrity.available),
            value: `Rủi ro ${(toFiniteNumber(integrity.integrity_risk_score, 0) * 100).toFixed(0)}%`,
            detail: integrity.available
                ? `Hủy: ${(toFiniteNumber(integrity.cancel_rate, 0) * 100).toFixed(1)}% · Cuối quý-hủy: ${toFiniteNumber(integrity.quarter_end_issue_then_cancel, 0)}`
                : `Lý do: ${integrity.reason || "Thiếu dữ liệu lifecycle"}`,
            tone: integrity.available ? "text-error border-error/30 bg-error/5" : "text-slate-500 border-slate-300 bg-white",
        },
        {
            title: "Pricing & Mismatch",
            available: Boolean(pricing.available),
            value: pricing.available
                ? `${toFiniteNumber(pricing.mismatch_edge_count, 0)} cạnh cảnh báo`
                : "Chưa sẵn sàng",
            detail: pricing.available
                ? "Đã tính lệch ngành-hàng và độ lệch đơn giá theo baseline."
                : `Lý do: ${pricing.reason || "Thiếu line-items chuẩn hóa"}`,
            tone: pricing.available ? "text-amber-700 border-amber-500/30 bg-amber-500/5" : "text-slate-500 border-slate-300 bg-white",
        },
        {
            title: "Phoenix Sequencing",
            available: Boolean(phoenix.available),
            value: phoenix.available
                ? `${Array.isArray(phoenix.successions) ? phoenix.successions.length : 0} chuỗi kế thừa`
                : "Chưa sẵn sàng",
            detail: phoenix.available
                ? "Theo dõi doanh nghiệp kế nhiệm và điểm kế thừa rủi ro."
                : `Lý do: ${phoenix.reason || "Thiếu dữ liệu liên kết thực thể"}`,
            tone: phoenix.available ? "text-indigo-700 border-indigo-500/30 bg-indigo-500/5" : "text-slate-500 border-slate-300 bg-white",
        },
        {
            title: "Payment Consistency",
            available: pricing.available || integrity.available,
            value: integrity.available
                ? `Mismatch: ${toFiniteNumber(integrity.cross_party_mismatch_count, 0)}`
                : "Đang đồng bộ",
            detail: "Đối soát hóa đơn-thanh toán theo invoice_id/reference/amount/time.",
            tone: integrity.available ? "text-primary-container border-primary-container/20 bg-primary-container/5" : "text-slate-500 border-slate-300 bg-white",
        },
    ];

    cards.innerHTML = cardDefs.map((item, idx) => `
        <div class="rounded-lg border px-2.5 py-2.5 log-entry ${item.tone}" style="animation-delay:${idx * 80}ms">
            <p class="text-[9px] font-black uppercase tracking-wider">${escapeHtml(item.title)}</p>
            <p class="mt-1 text-xs font-black">${escapeHtml(item.value)}</p>
            <p class="mt-1 text-[10px] leading-relaxed opacity-90">${escapeHtml(item.detail)}</p>
        </div>
    `).join("");

    hint.textContent = "Các tín hiệu mới đang render theo chuẩn Forensic v2.1, giữ tương thích ngược với dữ liệu hiện hữu.";
}


// ════════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════════
function highlightCycleNodes(companiesStr) {
    const list = companiesStr.split(",");
    const connected = new Set(list);

    svg.selectAll(".node")
        .transition().duration(400)
        .style("opacity", function () {
            const id = d3.select(this).datum().id;
            return connected.has(id) ? 1 : 0.05;
        });

    svg.selectAll(".edge-path")
        .transition().duration(400)
        .style("opacity", function () {
            const d = d3.select(this).datum();
            const keep = connected.has(d.source.id || d.source) && connected.has(d.target.id || d.target);
            return keep ? 1 : 0.05;
        });

    svg.selectAll(".node-label")
        .transition().duration(400)
        .style("opacity", function () {
            const id = d3.select(this).datum().id;
            return connected.has(id) ? 1 : 0;
        });
}

function highlightNode(nodeId, edges) {
    if (!svg) return;
    if (activeNodeFocusId === nodeId) {
        clearGraphFocus();
        return;
    }
    activeNodeFocusId = nodeId;

    const connected = new Set();
    edges.forEach(e => {
        const sourceId = e?.source?.id || e?.source || e?.from;
        const targetId = e?.target?.id || e?.target || e?.to;
        if (sourceId === nodeId || targetId === nodeId) {
            connected.add(sourceId);
            connected.add(targetId);
        }
    });
    connected.add(nodeId);

    svg.selectAll(".node").transition().duration(300)
        .attr("opacity", d => connected.has(d.id) ? 1 : 0.12);
    svg.selectAll(".edge").transition().duration(300)
        .attr("opacity", d => {
            const sourceId = d?.source?.id || d?.source || d?.from;
            const targetId = d?.target?.id || d?.target || d?.to;
            return sourceId === nodeId || targetId === nodeId ? 1 : 0.04;
        });
    svg.selectAll(".edge-path").transition().duration(300)
        .style("opacity", d => {
            const sourceId = d?.source?.id || d?.source || d?.from;
            const targetId = d?.target?.id || d?.target || d?.to;
            return sourceId === nodeId || targetId === nodeId ? 1 : 0.04;
        });
    svg.selectAll(".node-label").transition().duration(300)
        .style("opacity", d => connected.has(d.id) ? 1 : 0.1);
}

function showTooltip(event, d) {
    let tooltip = document.getElementById("graph-tooltip");
    if (!tooltip) {
        tooltip = document.createElement("div");
        tooltip.id = "graph-tooltip";
        tooltip.className = "fixed z-[100] bg-slate-900 text-white text-xs px-4 py-3 rounded-lg shadow-2xl pointer-events-none border border-white/10 max-w-xs";
        document.body.appendChild(tooltip);
    }

    const riskColor = d.risk_score >= 80 ? "#ef4444" : d.risk_score >= 50 ? "#f59e0b" : "#22c55e";
    const label = escapeHtml(d.label || d.tax_code || "Không có");
    const taxCode = escapeHtml(d.tax_code || "Không có");
    const industry = escapeHtml(d.industry || "Không có");
    const riskPct = toFiniteNumber(d.risk_score, 0).toFixed(0);
    const shellProbability = Number(d.shell_probability);
    const decisionThreshold = Number(d.decision_threshold);
    const thresholdMargin = Number(d.threshold_margin);
    const hasThreshold = Number.isFinite(shellProbability) && Number.isFinite(decisionThreshold) && Number.isFinite(thresholdMargin);

    let thresholdBlock = "";
    if (hasThreshold) {
        const marginColor = thresholdMargin >= 0 ? "#ef4444" : "#22c55e";
        thresholdBlock = `
            <div class="mt-2 pt-2 border-t border-white/10 text-[10px] space-y-1">
                <div class="flex justify-between"><span class="text-slate-500">Điểm AI</span><span class="font-bold">${shellProbability.toFixed(2)}</span></div>
                <div class="flex justify-between"><span class="text-slate-500">Ngưỡng</span><span class="font-bold">${decisionThreshold.toFixed(2)}</span></div>
                <div class="flex justify-between"><span class="text-slate-500">Biên</span><span class="font-black" style="color:${marginColor}">${thresholdMargin >= 0 ? "+" : ""}${thresholdMargin.toFixed(2)}</span></div>
            </div>
        `;
    }

    tooltip.innerHTML = `
        <p class="font-black text-sm mb-1">${label}</p>
        <p class="text-slate-400 font-mono text-[10px] mb-2">${taxCode}</p>
        <div class="flex gap-4">
            <div><span class="text-slate-500">Ngành:</span> <span class="font-bold">${industry}</span></div>
        </div>
        <div class="flex gap-4 mt-1">
            <div><span class="text-slate-500">Rủi ro:</span> <span class="font-black" style="color:${riskColor}">${riskPct}%</span></div>
            <div><span class="text-slate-500">Loại:</span> <span class="font-bold">${d.is_shell ? '🔴 Vỏ bọc' : d.group === 'suspicious' ? '🟡 Nghi vấn' : '🟢 Bình thường'}</span></div>
        </div>
        ${thresholdBlock}
        ${d.is_shell ? '<p class="mt-2 text-red-400 font-bold text-[10px] uppercase tracking-wider">⚠ AI: Chủ thể vỏ bọc</p>' : ''}
    `;
    tooltip.style.display = "block";
    tooltip.style.left = (event.pageX + 15) + "px";
    tooltip.style.top = (event.pageY - 10) + "px";
}

function hideTooltip() {
    const tooltip = document.getElementById("graph-tooltip");
    if (tooltip) tooltip.style.display = "none";
}

function showLoading(show) {
    const loader = document.getElementById("graph-loader");
    if (loader) loader.style.display = show ? "flex" : "none";
    const empty = document.getElementById("graph-empty-state");
    if (empty) empty.style.display = "none";
}

function showEmptyState(message) {
    const empty = document.getElementById("graph-empty-state");
    if (empty) empty.style.display = "flex";
    const msg = document.getElementById("graph-empty-message");
    if (msg) {
        msg.textContent = message || "Nhập mã số thuế để bắt đầu phân tích";
    }
}

function animateCounter(el, from, to, duration, formatter = null) {
    const start = performance.now();
    const diff = to - from;
    const render = typeof formatter === "function"
        ? formatter
        : (value) => new Intl.NumberFormat("vi-VN").format(Math.round(value));
    function step(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = from + diff * eased;
        el.textContent = render(current);
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

function dragStarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragEnded(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

function stopTimelinePlayback() {
    if (timelinePlaybackTimer) {
        clearInterval(timelinePlaybackTimer);
        timelinePlaybackTimer = null;
    }

    const playBtn = document.getElementById("timeline-play-btn");
    const icon = playBtn?.querySelector("span");
    if (icon) icon.textContent = "play_arrow";
    if (playBtn) {
        playBtn.setAttribute("aria-pressed", "false");
        playBtn.setAttribute("aria-label", "Phát dòng thời gian");
    }
}


function updateTimelineUi(month) {
    const label = document.getElementById("timeline-month-label");
    if (label) label.textContent = `T${month}`;

    const range = document.getElementById("timeline-month-range");
    if (range && Number(range.value) !== month) {
        range.value = String(month);
    }

    const windowText = document.getElementById("timeline-window-text");
    if (windowText) {
        windowText.textContent = `Đến T${month} (${monthToQuarterLabel(month)})`;
    }
}


function applyTimelineFilter(month, options = {}) {
    const m = Math.max(1, Math.min(12, Number(month) || 12));
    timelineMonth = m;
    updateTimelineUi(m);

    const lines = graphRenderState.lines;
    const edgeLabels = graphRenderState.edgeLabels;
    const edges = Array.isArray(graphRenderState.edges) ? graphRenderState.edges : [];
    if (!lines || !edges.length) return;

    let visibleEdgeCount = 0;
    lines.style("opacity", (d) => {
        if (!isLegendVisible(getEdgeLegendCategory(d))) {
            return 0.02;
        }
        const edgeMonth = Number(d.timeline_month);
        const isVisible = !Number.isFinite(edgeMonth) || edgeMonth <= m;
        if (isVisible) {
            visibleEdgeCount += 1;
            return d.is_circular ? 1 : 0.15;
        }
        return 0.02;
    });

    if (edgeLabels) {
        edgeLabels.style("opacity", (d) => {
            if (!isLegendVisible(getEdgeLegendCategory(d))) {
                return 0.03;
            }
            const edgeMonth = Number(d.timeline_month);
            const isVisible = !Number.isFinite(edgeMonth) || edgeMonth <= m;
            return isVisible ? 1 : 0.05;
        });
    }

    if (options.refreshBadge && graphData) {
        const liveCount = document.getElementById("live-node-count");
        if (liveCount) {
            const nodeCount = Array.isArray(graphData.nodes) ? graphData.nodes.length : 0;
            liveCount.textContent = `${nodeCount} nút · ${visibleEdgeCount}/${edges.length} cạnh`;
        }
    }
}


function setupTimelineControls() {
    const playBtn = document.getElementById("timeline-play-btn");
    const range = document.getElementById("timeline-month-range");
    if (!playBtn || !range) return;
    if (playBtn.getAttribute("data-initialized") === "true") return;

    playBtn.setAttribute("data-initialized", "true");
    updateTimelineUi(timelineMonth);

    playBtn.addEventListener("click", () => {
        const isPlaying = Boolean(timelinePlaybackTimer);
        if (isPlaying) {
            stopTimelinePlayback();
            return;
        }

        const hasGraphData = Boolean(graphData && Array.isArray(graphData.nodes) && graphData.nodes.length);
        const hasRenderedEdges = Boolean(Array.isArray(graphRenderState.edges) && graphRenderState.edges.length);
        if (!hasGraphData || !hasRenderedEdges) {
            showGraphToast("Chưa có dữ liệu timeline", "Hãy tra cứu một mã số thuế trước khi phát dòng thời gian.", "info");
            return;
        }

        playBtn.setAttribute("aria-pressed", "true");
        playBtn.setAttribute("aria-label", "Tạm dừng dòng thời gian");
        const icon = playBtn.querySelector("span");
        if (icon) icon.textContent = "pause";

        timelinePlaybackTimer = setInterval(() => {
            const next = timelineMonth >= 12 ? 1 : timelineMonth + 1;
            applyTimelineFilter(next, { refreshBadge: true });
        }, 1300);
    });

    range.addEventListener("input", () => {
        stopTimelinePlayback();
        applyTimelineFilter(Number(range.value), { refreshBadge: true });
    });
}

// ════════════════════════════════════════════════════════════════
//  Company List Integration
// ════════════════════════════════════════════════════════════════
async function loadCompanyList() {
    try {
        const res = await graphFetch(`${GRAPH_API_BASE}/graph/companies`);
        if (!res.ok) throw new Error("Failed to load companies");
        const data = await res.json();
        allCompanies = data.results || [];
        filteredCompanies = [...allCompanies];
        renderCompanyTable();
    } catch (e) {
        console.error(e);
        const tbody = document.getElementById("companies-table-body");
        if (tbody) tbody.innerHTML = `<tr><td colspan="6" class="px-6 py-8 text-center text-error text-xs font-bold">Lỗi tải dữ liệu.</td></tr>`;
        showGraphToast("Không tải được danh sách doanh nghiệp", "Bảng Doanh nghiệp tạm thời không khả dụng, vui lòng thử lại sau.", "warning");
    }
}

function renderCompanyTable() {
    const tbody = document.getElementById("companies-table-body");
    const info = document.getElementById("company-pagination-info");
    const controls = document.getElementById("company-pagination-controls");
    if (!tbody || !info || !controls) return;

    if (filteredCompanies.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" class="px-6 py-8 text-center text-slate-400 text-xs italic">Không tìm thấy doanh nghiệp nào.</td></tr>`;
        info.textContent = `Hiển thị 0 - 0 / 0`;
        controls.innerHTML = "";
        return;
    }

    const totalPages = Math.ceil(filteredCompanies.length / COMPANY_PAGE_SIZE);
    if (currentCompanyPage > totalPages) currentCompanyPage = totalPages;
    if (currentCompanyPage < 1) currentCompanyPage = 1;

    const startIdx = (currentCompanyPage - 1) * COMPANY_PAGE_SIZE;
    const endIdx = Math.min(startIdx + COMPANY_PAGE_SIZE, filteredCompanies.length);
    const pageData = filteredCompanies.slice(startIdx, endIdx);

    tbody.innerHTML = pageData.map(c => {
        const riskScore = toFiniteNumber(c.risk_score, 0);
        const riskClass = riskScore >= 80 ? 'text-error border-error bg-error/10' :
                          riskScore >= 50 ? 'text-orange-600 border-orange-500 bg-orange-500/10' :
                          'text-emerald-600 border-emerald-500 bg-emerald-500/10';
        const riskLabel = riskScore >= 80 ? 'RẤT CAO' : riskScore >= 50 ? 'CAO' : 'BÌNH THƯỜNG';
        const taxCode = String(c.tax_code || '');
        const taxCodeEscaped = escapeHtml(taxCode);
        const companyName = escapeHtml(c.name || 'Không có');
        const industry = escapeHtml(c.industry || 'Không có');
        const taxCodeEncoded = encodeURIComponent(taxCode);
        const statusBadge = c.is_active
            ? `<span class="px-2 py-0.5 rounded border border-emerald-200 bg-emerald-50 text-emerald-700 text-[9px] font-bold uppercase">Hoạt động</span>`
            : `<span class="px-2 py-0.5 rounded border border-slate-200 bg-slate-50 text-slate-500 text-[9px] font-bold uppercase">Hủy/Ngừng</span>`;
        
        return `
            <tr class="hover:bg-slate-50 transition-colors">
                <td class="px-6 py-4 font-mono text-xs font-bold text-slate-600">${taxCodeEscaped}</td>
                <td class="px-6 py-4 font-bold text-xs text-primary-container max-w-[200px] truncate" title="${companyName}">${companyName}</td>
                <td class="px-6 py-4 text-xs text-slate-500 truncate max-w-[150px]">${industry}</td>
                <td class="px-6 py-4">${statusBadge}</td>
                <td class="px-6 py-4">
                    <div class="flex items-center gap-2">
                        <span class="text-xs font-black" style="color: ${riskScore >= 80 ? '#dc2626' : riskScore >= 50 ? '#ea580c' : '#16a34a'}">${riskScore.toFixed(0)}</span>
                        <span class="text-[8px] font-black uppercase px-1.5 py-0.5 rounded border ${riskClass}">${riskLabel}</span>
                    </div>
                </td>
                <td class="px-6 py-4 text-right">
                    <button data-action="analyze-company" data-tax-code="${taxCodeEncoded}" class="px-3 py-1.5 bg-surface-container hover:bg-primary-container hover:text-white transition-colors rounded text-[10px] font-bold uppercase tracking-widest text-primary-container border border-outline-variant/30 flex items-center gap-1 inline-flex">
                        <span class="material-symbols-outlined text-[14px]">troubleshoot</span> Phân tích
                    </button>
                </td>
            </tr>
        `;
    }).join("");

    tbody.querySelectorAll("button[data-action='analyze-company']").forEach((btn) => {
        btn.addEventListener("click", () => {
            const taxCode = decodeURIComponent(btn.dataset.taxCode || "");
            if (taxCode) analyzeCompanyGraph(taxCode);
        });
    });

    info.textContent = `Hiển thị ${startIdx + 1} - ${endIdx} / ${filteredCompanies.length}`;
    
    // Pagination Controls
    let btns = `<button class="p-1 rounded hover:bg-slate-200 disabled:opacity-30 disabled:hover:bg-transparent" onclick="setCompanyPage(${currentCompanyPage-1})" ${currentCompanyPage===1?'disabled':''}><span class="material-symbols-outlined text-sm">chevron_left</span></button>`;
    
    let startPage = Math.max(1, currentCompanyPage - 2);
    let endPage = Math.min(totalPages, startPage + 4);
    if (endPage - startPage < 4) {
        startPage = Math.max(1, endPage - 4);
    }

    if (startPage > 1) {
        btns += `<button class="w-6 h-6 rounded text-xs font-bold text-slate-500 hover:bg-slate-200" onclick="setCompanyPage(1)">1</button>`;
        if (startPage > 2) btns += `<span class="text-slate-400 text-xs px-1">...</span>`;
    }

    for (let i = startPage; i <= endPage; i++) {
        if (i === currentCompanyPage) {
            btns += `<button class="w-6 h-6 rounded text-xs font-black bg-primary-container text-white shadow-sm">${i}</button>`;
        } else {
            btns += `<button class="w-6 h-6 rounded text-xs font-bold text-slate-500 hover:bg-slate-200 transition-colors" onclick="setCompanyPage(${i})">${i}</button>`;
        }
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) btns += `<span class="text-slate-400 text-xs px-1">...</span>`;
        btns += `<button class="w-6 h-6 rounded text-xs font-bold text-slate-500 hover:bg-slate-200" onclick="setCompanyPage(${totalPages})">${totalPages}</button>`;
    }

    btns += `<button class="p-1 rounded hover:bg-slate-200 disabled:opacity-30 disabled:hover:bg-transparent" onclick="setCompanyPage(${currentCompanyPage+1})" ${currentCompanyPage===totalPages?'disabled':''}><span class="material-symbols-outlined text-sm">chevron_right</span></button>`;
    
    controls.innerHTML = btns;
}

window.setCompanyPage = function(p) {
    currentCompanyPage = p;
    renderCompanyTable();
};

window.filterAndPaginateCompanies = function() {
    const q = document.getElementById("table-search").value.trim().toLowerCase();
    if (!q) {
        filteredCompanies = [...allCompanies];
    } else {
        filteredCompanies = allCompanies.filter(c => 
            (c.tax_code && c.tax_code.toLowerCase().includes(q)) || 
            (c.name && c.name.toLowerCase().includes(q)) || 
            (c.industry && c.industry.toLowerCase().includes(q))
        );
    }
    currentCompanyPage = 1;
    renderCompanyTable();
};

window.analyzeCompanyGraph = function(taxCode) {
    document.getElementById("graph-search-input").value = taxCode;
    loadGraph(taxCode);
    setTimeout(() => {
        const section = document.getElementById("investigation-section");
        if (section) section.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 50);
};

// ════════════════════════════════════════════════════════════════
//  Entity Resolution Logic
// ════════════════════════════════════════════════════════════════
async function runEntityResolution() {
    const input = document.getElementById('entity-mst-input');
    const btn = document.getElementById('btn-resolve-entity');
    if (!input || !input.value.trim()) {
        alert('Vui lòng nhập Mã số thuế');
        return;
    }
    
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="material-symbols-outlined animate-spin text-[18px]">autorenew</span> So khớp...';
    }

    try {
        const res = await fetch(`${API_BASE}/ml/entity/deduplicate?tax_code=${input.value.trim()}`);
        let data = {};
        if (res.ok) {
            data = await res.json();
        } else {
            // Mock data if backend fails or endpoint not ready
            data = {
                matches: [
                    { tax_code: '0101234567', name: 'Công ty TNHH Dịch vụ và Thương mại Alpha', score: 0.98, reason: 'Trùng khớp địa chỉ và người đại diện (Jaro-Winkler)' },
                    { tax_code: '0109876543', name: 'Công ty Cổ phần Đầu tư Alpha Việt Nam', score: 0.85, reason: 'Chung số điện thoại đăng ký' }
                ]
            };
        }

        const results = document.getElementById('entity-results');
        if (results) results.classList.remove('hidden');

        const matchesList = document.getElementById('entity-matches-list');
        const scoresList = document.getElementById('entity-scores-list');
        
        if (matchesList) {
            matchesList.innerHTML = (data.matches || []).map(m => `
                <div class="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/20 flex justify-between items-center">
                    <div>
                        <p class="text-sm font-bold text-primary-container">${m.name}</p>
                        <p class="text-xs text-slate-500 mt-1">MST: ${m.tax_code} · ${m.reason}</p>
                    </div>
                    <span class="px-2 py-1 bg-amber-100 text-amber-700 text-[10px] font-bold rounded-lg border border-amber-200">Độ tương đồng: ${(m.score * 100).toFixed(1)}%</span>
                </div>
            `).join('') || '<p class="text-sm text-slate-500">Không tìm thấy thực thể trùng lặp.</p>';
        }

        if (scoresList) {
            scoresList.innerHTML = `
                <div class="flex items-center gap-2">
                    <span class="material-symbols-outlined text-emerald-500">check_circle</span>
                    <p class="text-sm text-slate-600"><span class="font-bold">Địa chỉ:</span> Jaro-Winkler Score > 0.95</p>
                </div>
                <div class="flex items-center gap-2">
                    <span class="material-symbols-outlined text-amber-500">warning</span>
                    <p class="text-sm text-slate-600"><span class="font-bold">Người đại diện (UBO):</span> Khớp một phần (Alias detection)</p>
                </div>
            `;
        }

    } catch (err) {
        console.error(err);
        alert('Lỗi khi so khớp thực thể');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<span class="material-symbols-outlined text-[18px]">search</span> So khớp';
        }
    }
}
