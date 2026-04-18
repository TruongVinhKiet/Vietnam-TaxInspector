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
let timelineMonth = 12;
let timelinePlaybackTimer = null;
let graphRenderState = {
    edges: [],
    lines: null,
    edgeLabels: null,
};
let graphSplitTriggerGate = null;

let allCompanies = [];
let filteredCompanies = [];
let currentCompanyPage = 1;
const COMPANY_PAGE_SIZE = 10;


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
                renderForensicPanel(graphData, forensicIntel);
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
    try {
        const res = await graphFetch(url);
        if (!res.ok) return null;
        return await res.json();
    } catch (err) {
        console.warn("Graph supplemental API warning:", err);
        return null;
    }
}


async function loadForensicIntel(taxCode, depth = 2) {
    const ownershipParams = new URLSearchParams();
    if (taxCode) ownershipParams.set("tax_code", taxCode);

    const ringParams = new URLSearchParams();
    if (taxCode) ringParams.set("tax_code", taxCode);
    ringParams.set("depth", String(Math.max(1, Math.min(4, Number(depth) || 2))));

    const ownershipUrl = `${GRAPH_API_BASE}/graph/ownership${ownershipParams.toString() ? `?${ownershipParams.toString()}` : ""}`;
    const ringUrl = `${GRAPH_API_BASE}/graph/ring-scoring?${ringParams.toString()}`;

    const [ownershipPayload, ringPayload] = await Promise.all([
        safeGraphJson(ownershipUrl),
        safeGraphJson(ringUrl),
    ]);

    return {
        ownership: ownershipPayload && typeof ownershipPayload === "object" ? ownershipPayload : null,
        ring: ringPayload && typeof ringPayload === "object" ? ringPayload : null,
        fetched_at: new Date().toISOString(),
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
        } else if (event.key === "Escape") {
            clearGraphFocus();
        }
    });
}


function switchWorkbenchMode(mode, options = {}) {
    const validModes = new Set(["graph", "forensic", "companies"]);
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
    if (!investigationSection || !companiesSection) return;

    if (nextMode !== "graph") {
        stopTimelinePlayback();
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
    const edgesLayer = zoomGroup.select(".edges-layer");
    const nodesLayer = zoomGroup.select(".nodes-layer");
    const labelsLayer = zoomGroup.select(".labels-layer");

    // Clear previous
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

    simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(edges).id(d => d.id).distance(180).strength(0.6))
        .force("charge", d3.forceManyBody().strength(-800))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(60).iterations(4))
        .alphaDecay(0.05);

    // ── Edges ──
    const edgeGroups = edgesLayer.selectAll("g.edge")
        .data(edges, d => d.invoice_number || `${d.from}-${d.to}-${d.linkIndex}`)
        .join("g")
        .attr("class", "edge");

    const lines = edgeGroups.append("path")
        .attr("class", d => d.is_circular ? "edge-path circular" : "edge-path normal")
        .attr("fill", "none")
        .attr("stroke", d => d.is_circular ? "#ba1a1a" : "#334155")
        .attr("stroke-width", d => d.is_circular ? 2 : 1)
        .attr("stroke-dasharray", d => d.is_circular ? "8 4" : "none")
        .attr("marker-end", d => d.is_circular ? "url(#arrow-fraud)" : "url(#arrow-normal)")
        .attr("opacity", d => d.is_circular ? 1 : 0.15);

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
        .attr("class", "node")
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
            if (d.is_shell) return "#ba1a1a"; // Exact mock red
            if (d.group === "suspicious") return "#1e293b"; // slate-800
            return "#002147"; // primary-container
        })
        .attr("stroke", d => {
            if (d.is_shell) return "rgba(255,255,255,0.4)";
            if (d.group === "suspicious") return "rgba(255,255,255,0.15)";
            return "rgba(255,255,255,0.3)";
        })
        .attr("stroke-width", 1)
        .attr("class", "node-rect")
        .style("filter", d => d.is_shell ? "drop-shadow(0 10px 15px rgb(186 26 26 / 0.5))" : "drop-shadow(0 4px 6px rgb(0 0 0 / 0.3))");
        
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
        .attr("fill", d => d.group === "suspicious" ? "#cbd5e1" : "#ffffff")
        .attr("font-size", d => d.is_shell ? "18px" : "16px")
        .style("font-family", "Material Symbols Outlined")
        .text(d => iconMap[d.group] || "factory");

    // Node labels inside the rect
    nodeGroups.append("text")
        .attr("class", "node-label")
        .attr("text-anchor", "start")
        .attr("x", -18)
        .attr("y", 4)
        .attr("fill", "#ffffff")
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
            .attr("stroke", d.is_shell ? "rgba(255,255,255,0.4)" : d.group === "suspicious" ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.3)");
    }).on("click", function(event, d) {
        // Highlight connected edges
        highlightNode(d.id, edges);
    });

    // ── Tick ──
    simulation.on("tick", () => {
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

    applyTimelineFilter(timelineMonth, { refreshBadge: true });

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


function renderForensicIntelSummary(forensicIntel) {
    const strip = document.getElementById("forensic-intel-strip");
    if (!strip) return;

    const ringPayload = forensicIntel && typeof forensicIntel.ring === "object" ? forensicIntel.ring : null;
    const ownershipPayload = forensicIntel && typeof forensicIntel.ownership === "object" ? forensicIntel.ownership : null;
    const ownershipSummary = ownershipPayload && typeof ownershipPayload.summary === "object" ? ownershipPayload.summary : {};

    const ringTotal = toFiniteNumber(ringPayload?.total, 0);
    const ringCritical = toFiniteNumber(ringPayload?.critical_count, 0);
    const ownershipClusters = toFiniteNumber(ownershipSummary.total_clusters, 0);
    const crossTrades = toFiniteNumber(ownershipSummary.total_cross_trades, 0);
    const commonControllers = toFiniteNumber(ownershipSummary.total_common_controllers, 0);

    const hasAnySignal = ringTotal > 0 || ringCritical > 0 || ownershipClusters > 0 || crossTrades > 0 || commonControllers > 0;
    if (!hasAnySignal) {
        strip.classList.add("hidden");
        return;
    }

    strip.classList.remove("hidden");
    setElementText("forensic-ring-total", String(ringTotal));
    setElementText("forensic-ring-critical", String(ringCritical));
    setElementText("forensic-ownership-clusters", String(ownershipClusters));
    setElementText("forensic-cross-trades", String(crossTrades));

    const updatedEl = document.getElementById("forensic-intel-updated");
    if (updatedEl) {
        const ts = formatShortTimestamp(forensicIntel?.fetched_at);
        updatedEl.textContent = `Nguồn sở hữu/vòng đã đồng bộ${ts !== "--" ? ` lúc ${ts}` : ""} · Bộ điều phối: ${commonControllers}`;
    }
}


function buildIntegratedForensicLogs(data, forensicIntel) {
    const baseLogsRaw = Array.isArray(data?.logs) ? data.logs : [];
    const baseLogs = baseLogsRaw
        .filter((item) => item && typeof item === "object")
        .map((log) => ({
            timestamp: String(log.timestamp || formatForensicTimestamp()),
            severity: String(log.severity || "low").toLowerCase(),
            title: String(log.title || "Cập nhật điều tra"),
            description: String(log.description || "Không có mô tả bổ sung."),
        }));

    const ringPayload = forensicIntel && typeof forensicIntel.ring === "object" ? forensicIntel.ring : null;
    const ownershipPayload = forensicIntel && typeof forensicIntel.ownership === "object" ? forensicIntel.ownership : null;
    const ownershipSummary = ownershipPayload && typeof ownershipPayload.summary === "object" ? ownershipPayload.summary : {};

    const integratedLogs = [];
    const nowTs = formatForensicTimestamp(forensicIntel?.fetched_at);

    const ringTotal = toFiniteNumber(ringPayload?.total, 0);
    const ringCritical = toFiniteNumber(ringPayload?.critical_count, 0);
    if (ringTotal > 0) {
        integratedLogs.push({
            timestamp: nowTs,
            severity: ringCritical > 0 ? "high" : "medium",
            title: `Phát hiện ${ringTotal} vòng giao dịch nghi vấn`,
            description: `Hệ thống Ring Scoring ghi nhận ${ringCritical} vòng mức critical; đề nghị ưu tiên đối chiếu các chuỗi có ring score cao.`,
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
    if (ownershipClusters > 0 || commonControllers > 0) {
        integratedLogs.push({
            timestamp: nowTs,
            severity: "medium",
            title: `Mạng sở hữu có ${ownershipClusters} cụm và ${commonControllers} đầu mối chung`,
            description: "Đề nghị rà soát luồng hóa đơn phát sinh trong các cụm sở hữu để loại trừ kịch bản điều phối giao dịch nội bộ.",
        });
    }

    const mergedLogs = [...integratedLogs, ...baseLogs]
        .filter((log) => log.title && log.description)
        .slice(0, 30);

    return mergedLogs;
}


function buildIntegratedEvidencePaths(data, forensicIntel) {
    const basePaths = Array.isArray(data?.evidence_paths) ? data.evidence_paths : [];
    const integratedPaths = [];

    const ringPayload = forensicIntel && typeof forensicIntel.ring === "object" ? forensicIntel.ring : null;
    const ringList = Array.isArray(ringPayload?.rings) ? ringPayload.rings : [];
    ringList.slice(0, 6).forEach((ring, index) => {
        const nodes = Array.isArray(ring?.nodes) ? ring.nodes.map((n) => String(n || "")).filter(Boolean) : [];
        if (nodes.length < 2) return;

        const totalAmount = toFiniteNumber(ring.total_amount, 0);
        const perHopAmount = nodes.length ? totalAmount / nodes.length : totalAmount;
        const ringScore = toFiniteNumber(ring.ring_score, 0);
        const spanDays = Number.isFinite(Number(ring.time_span_days)) ? `${Number(ring.time_span_days)} ngày` : "Không có";

        const hops = nodes.map((fromNode, hopIdx) => {
            const toNode = nodes[(hopIdx + 1) % nodes.length];
            return {
                from: fromNode,
                to: toNode,
                amount_formatted: formatVndCompact(perHopAmount),
                date: spanDays,
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
    crossTrades.slice(0, 6).forEach((trade, index) => {
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
                    date: "Không có",
                    fraud_probability: Math.min(1, ownershipPercent / 100),
                },
            ],
        });
    });

    return [...integratedPaths, ...basePaths];
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
        animateCounter(totalEl, 0, amount, 1500);
    }

    // Invoice count
    const countEl = document.getElementById("suspicious-invoice-count");
    if (countEl) {
        const totalInv = toFiniteNumber(payload.total_suspicious_invoices, 0);
        countEl.textContent = `Bao gồm ${totalInv} hóa đơn không phát sinh hàng hóa thật được luân chuyển.`;
    }

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
            pathBadge.textContent = String(evidencePaths.length);
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
                
                const hopsHTML = hops.length ? hops.map((h) => `
                    <div class="relative pl-6 pb-4 border-l-2 border-slate-200 last:border-transparent last:pb-0">
                        <div class="absolute -left-[5px] top-1 w-2 h-2 rounded-full bg-primary-container z-10"></div>
                        <div class="bg-surface-container-low p-2 rounded-lg border border-outline-variant/30 text-[10px]">
                            <div class="flex justify-between font-bold text-primary-container mb-1">
                                <span>${escapeHtml(String(h.from || "").substring(0, 6))}... → ${escapeHtml(String(h.to || "").substring(0, 6))}...</span>
                                <span class="text-error">${escapeHtml(h.amount_formatted || formatVndCompact(h.amount || 0))}</span>
                            </div>
                            <div class="flex justify-between text-slate-500 font-mono">
                                <span>Ngày: ${escapeHtml(h.date || "")}</span>
                                <span>Xác suất: ${toFiniteNumber(h.fraud_probability, 0).toFixed(2)}</span>
                            </div>
                        </div>
                    </div>
                `).join("") : `<p class="text-[10px] text-slate-400 italic">Chuỗi này chưa có chi tiết giao dịch theo từng chặng.</p>`;

                return `
                    <div class="border border-outline-variant/30 rounded-xl p-3 mb-4 log-entry" style="animation: fadeSlideIn 0.4s ease ${pIdx * 0.1}s both;">
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-xs font-black text-primary-container">${pathId}</span>
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
        }
    }

    setupForensicTabs();
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
    const connected = new Set();
    edges.forEach(e => {
        if (e.from === nodeId || e.to === nodeId) {
            connected.add(e.from);
            connected.add(e.to);
        }
    });

    svg.selectAll(".node").transition().duration(300)
        .attr("opacity", d => connected.has(d.id) || d.id === nodeId ? 1 : 0.15);
    svg.selectAll(".edge").transition().duration(300)
        .attr("opacity", d => d.from === nodeId || d.to === nodeId ? 1 : 0.05);

    // Reset after 3s
    setTimeout(() => {
        svg.selectAll(".node").transition().duration(500).attr("opacity", 1);
        svg.selectAll(".edge").transition().duration(500).attr("opacity", 1);
        setTimeout(() => applyTimelineFilter(timelineMonth, { refreshBadge: true }), 520);
    }, 3000);
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

function animateCounter(el, from, to, duration) {
    const start = performance.now();
    const diff = to - from;
    function step(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = from + diff * eased;
        el.textContent = new Intl.NumberFormat("vi-VN").format(Math.round(current)) + " VNĐ";
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
