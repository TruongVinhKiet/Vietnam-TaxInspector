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

// ════════════════════════════════════════════════════════════════
//  State
// ════════════════════════════════════════════════════════════════
let graphData = null;
let simulation = null;
let svg = null;
let currentTaxCode = null;

let allCompanies = [];
let filteredCompanies = [];
let currentCompanyPage = 1;
const COMPANY_PAGE_SIZE = 10;

// ════════════════════════════════════════════════════════════════
//  Init
// ════════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
    initGraph();
    setupSearch();
    setupTimelineControls();

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
                const res = await fetch(`${GRAPH_API_BASE}/graph/search?q=${encodeURIComponent(q)}`);
                const data = await res.json();
                renderSearchDropdown(data.results || [], dropdown);
            } catch (e) { console.warn("Search error:", e); }
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
}

function renderSearchDropdown(results, dropdown) {
    if (!results.length) { dropdown.classList.add("hidden"); return; }
    dropdown.innerHTML = results.map(r => `
        <div class="px-4 py-2 hover:bg-slate-50 cursor-pointer flex justify-between items-center search-result-item"
             data-tax-code="${r.tax_code}">
            <div>
                <p class="text-sm font-bold text-primary-container">${r.name}</p>
                <p class="text-[10px] text-slate-400 font-mono">${r.tax_code} · ${r.industry || ''}</p>
            </div>
            <span class="text-xs font-bold ${r.risk_score >= 60 ? 'text-error' : 'text-emerald-600'}">${r.risk_score.toFixed(0)}%</span>
        </div>
    `).join("");
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
        .attr("refX", 28).attr("refY", 0)
        .attr("markerWidth", 6).attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", "#475569");

    defs.append("marker")
        .attr("id", "arrow-fraud")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 28).attr("refY", 0)
        .attr("markerWidth", 6).attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", "#dc2626");

    // Glow filter
    const glow = defs.append("filter").attr("id", "glow");
    glow.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "blur");
    glow.append("feMerge").selectAll("feMergeNode")
        .data(["blur", "SourceGraphic"]).enter()
        .append("feMergeNode").attr("in", d => d);

    // Red glow filter
    const redGlow = defs.append("filter").attr("id", "red-glow");
    redGlow.append("feGaussianBlur").attr("stdDeviation", "4").attr("result", "blur");
    redGlow.append("feFlood").attr("flood-color", "#dc2626").attr("flood-opacity", "0.6").attr("result", "color");
    redGlow.append("feComposite").attr("in", "color").attr("in2", "blur").attr("operator", "in").attr("result", "colored-blur");
    const redMerge = redGlow.append("feMerge");
    redMerge.append("feMergeNode").attr("in", "colored-blur");
    redMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Zoom
    const zoomGroup = svg.append("g").attr("class", "zoom-group");
    svg.call(d3.zoom()
        .scaleExtent([0.2, 5])
        .on("zoom", (event) => {
            zoomGroup.attr("transform", event.transform);
            // Update zoom badge
            const badge = document.getElementById("zoom-badge");
            if (badge) badge.textContent = `TRƯỜNG NHÌN: ${Math.round(event.transform.k * 100)}%`;
        })
    );

    // Layers
    zoomGroup.append("g").attr("class", "edges-layer");
    zoomGroup.append("g").attr("class", "nodes-layer");
    zoomGroup.append("g").attr("class", "labels-layer");
}

async function loadGraph(taxCode) {
    const section = document.getElementById("investigation-section");
    if (!taxCode) {
        if (section) {
            section.classList.add("opacity-0");
            setTimeout(() => section.classList.add("hidden"), 500);
        }
        return;
    }

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
        let url = `${GRAPH_API_BASE}/graph`;
        url += `?tax_code=${encodeURIComponent(taxCode)}&depth=2`;

        const res = await fetch(url, { credentials: "include" });
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        graphData = await res.json();

        renderGraph(graphData);
        renderForensicPanel(graphData);
        showLoading(false);

    } catch (err) {
        console.error("Graph load error:", err);
        showLoading(false);
        showEmptyState();
    }
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
        showEmptyState();
        return;
    }

    const nodes = data.nodes.map(n => ({ ...n }));
    
    // Compute link index for parallel edges
    const linkCounts = {};
    const edges = data.edges.map(e => {
        const key = e.from < e.to ? `${e.from}-${e.to}` : `${e.to}-${e.from}`;
        const index = linkCounts[key] || 0;
        linkCounts[key] = index + 1;
        return {
            ...e,
            source: e.from,
            target: e.to,
            linkIndex: index
        };
    });

    // ── Simulation ──
    if (simulation) simulation.stop();

    simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(edges).id(d => d.id).distance(120).strength(0.7))
        .force("charge", d3.forceManyBody().strength(-400))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(45))
        .alphaDecay(0.02);

    // ── Edges ──
    const edgeGroups = edgesLayer.selectAll("g.edge")
        .data(edges, d => d.invoice_number || `${d.from}-${d.to}-${d.linkIndex}`)
        .join("g")
        .attr("class", "edge");

    const lines = edgeGroups.append("path")
        .attr("class", d => d.is_circular ? "edge-path circular" : "edge-path normal")
        .attr("fill", "none")
        .attr("stroke", d => d.is_circular ? "#dc2626" : "#475569")
        .attr("stroke-width", d => d.is_circular ? 2.5 : 1.2)
        .attr("stroke-dasharray", d => d.is_circular ? "8 4" : "none")
        .attr("marker-end", d => d.is_circular ? "url(#arrow-fraud)" : "url(#arrow-normal)")
        .attr("opacity", d => d.is_circular ? 1 : 0.4);

    // Edge labels (amount)
    const edgeLabels = edgeGroups.append("text")
        .attr("class", "edge-label")
        .attr("text-anchor", "middle")
        .attr("dy", -8)
        .attr("fill", d => d.is_circular ? "#fca5a5" : "#64748b")
        .attr("font-size", "9px")
        .attr("font-weight", d => d.is_circular ? "700" : "400")
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

    // Node outer pulse ring (for suspicious/shell)
    nodeGroups.filter(d => d.is_shell || d.group === "suspicious")
        .append("circle")
        .attr("r", d => d.is_shell ? 32 : 26)
        .attr("fill", "none")
        .attr("stroke", d => d.is_shell ? "#dc2626" : "#f59e0b")
        .attr("stroke-width", 2)
        .attr("opacity", 0.3)
        .attr("filter", d => d.is_shell ? "url(#red-glow)" : "none")
        .attr("class", "pulse-ring")
        .style("animation", "nodeGlow 3s ease-in-out infinite");

    // Node main circle
    nodeGroups.append("circle")
        .attr("r", d => {
            if (d.is_shell) return 26;
            if (d.group === "suspicious") return 22;
            return 18;
        })
        .attr("fill", d => {
            if (d.is_shell) return "#991b1b";
            if (d.group === "suspicious") return "#b45309";
            return "#1e3a5f";
        })
        .attr("stroke", d => d.is_shell ? "#fca5a5" : "#94a3b8")
        .attr("stroke-width", d => d.is_shell ? 3 : 1.5)
        .attr("class", "node-circle");

    // Node icons (Material Symbols workaround: use text)
    const iconMap = {
        "shell": "🏢",
        "suspicious": "⚠️",
        "normal": "🏭",
    };
    nodeGroups.append("text")
        .attr("text-anchor", "middle")
        .attr("dy", "0.35em")
        .attr("font-size", d => d.is_shell ? "16px" : "12px")
        .text(d => iconMap[d.group] || "🏭");

    // Node labels
    labelsLayer.selectAll("text.node-label")
        .data(nodes, d => d.id)
        .join("text")
        .attr("class", "node-label")
        .attr("text-anchor", "middle")
        .attr("dy", d => (d.is_shell ? 40 : 32))
        .attr("fill", d => d.is_shell ? "#fca5a5" : "#94a3b8")
        .attr("font-size", "8px")
        .attr("font-weight", d => d.is_shell ? "800" : "600")
        .attr("font-family", "Inter, sans-serif")
        .attr("letter-spacing", "0.05em")
        .text(d => {
            const name = d.label || d.tax_code;
            return name.length > 18 ? name.substring(0, 18) + "…" : name;
        });

    // ── Tooltip ──
    nodeGroups.on("mouseenter", function(event, d) {
        showTooltip(event, d);
        d3.select(this).select(".node-circle")
            .transition().duration(200)
            .attr("r", (d.is_shell ? 30 : d.group === "suspicious" ? 26 : 22));
    }).on("mouseleave", function(event, d) {
        hideTooltip();
        d3.select(this).select(".node-circle")
            .transition().duration(200)
            .attr("r", (d.is_shell ? 26 : d.group === "suspicious" ? 22 : 18));
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
        labelsLayer.selectAll(".node-label")
            .attr("x", (d, i) => nodes[i]?.x || 0)
            .attr("y", (d, i) => nodes[i]?.y || 0);
    });

    // Update legend badge
    const liveCount = document.getElementById("live-node-count");
    if (liveCount) liveCount.textContent = `${nodes.length} nút · ${edges.length} cạnh`;
}


// ════════════════════════════════════════════════════════════════
//  Forensic Panel
// ════════════════════════════════════════════════════════════════
function renderForensicPanel(data) {
    // Total suspicious amount
    const totalEl = document.getElementById("total-suspicious-amount");
    if (totalEl) {
        const amount = data.total_suspicious_amount || 0;
        totalEl.textContent = new Intl.NumberFormat("vi-VN").format(amount) + " VNĐ";
        // Animate counter
        animateCounter(totalEl, 0, amount, 1500);
    }

    // Invoice count
    const countEl = document.getElementById("suspicious-invoice-count");
    if (countEl) {
        const totalInv = data.total_suspicious_invoices || 0;
        countEl.textContent = `Bao gồm ${totalInv} hóa đơn không phát sinh hàng hóa thật được luân chuyển.`;
    }

    // Severity badge
    const badge = document.getElementById("severity-badge");
    if (badge) {
        const amount = data.total_suspicious_amount || 0;
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
    if (logsContainer && data.logs) {
        logsContainer.innerHTML = data.logs.map((log, i) => {
            const severityColors = {
                critical: { border: "border-error", dot: "bg-error", text: "text-error", label: "NGHIÊM TRỌNG" },
                high:     { border: "border-amber-500", dot: "bg-amber-500", text: "text-amber-600", label: "CAO" },
                medium:   { border: "border-blue-500", dot: "bg-blue-500", text: "text-blue-600", label: "TRUNG BÌNH" },
                low:      { border: "border-emerald-500", dot: "bg-emerald-500", text: "text-emerald-600", label: "THẤP" },
            };
            const s = severityColors[log.severity] || severityColors.low;

            return `
                <div class="relative pl-4 ${s.border} border-l-2 log-entry" style="animation: fadeSlideIn 0.4s ease ${i * 0.15}s both;">
                    <div class="absolute -left-[5px] top-1 w-2 h-2 rounded-full ${s.dot}"></div>
                    <div class="flex justify-between items-start mb-1">
                        <p class="text-[10px] font-mono text-slate-500 font-bold">${log.timestamp}</p>
                        <p class="text-[10px] font-bold ${s.text} uppercase">${s.label}</p>
                    </div>
                    <p class="text-sm font-bold text-primary-container leading-tight">${log.title}</p>
                    <p class="text-xs text-on-surface-variant mt-1">${log.description}</p>
                </div>
            `;
        }).join("");
    }

    // Model status badge
    const modelBadge = document.getElementById("model-status-badge");
    if (modelBadge) {
        if (data.model_loaded) {
            modelBadge.innerHTML = `<div class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div><span>AI GAT Model Active${data.ensemble_active ? " (Ensemble)" : ""}</span>`;
        } else {
            modelBadge.innerHTML = `<div class="w-2 h-2 rounded-full bg-amber-500"></div><span>Heuristic Mode</span>`;
        }
    }

    // Evidence Paths Rendering
    const pathsContainer = document.getElementById("evidence-paths-container");
    const pathBadge = document.getElementById("path-badge");
    if (pathsContainer && data.evidence_paths) {
        if (data.evidence_paths.length > 0) {
            pathBadge.textContent = data.evidence_paths.length;
            pathBadge.classList.remove("hidden");
        } else {
            pathBadge.classList.add("hidden");
            pathsContainer.innerHTML = `
                <div class="flex items-center justify-center h-32">
                    <p class="text-xs text-slate-400 italic">Không phát hiện chuỗi quay vòng tuần hoàn.</p>
                </div>`;
        }

        const pathHTML = data.evidence_paths.map((p, pIdx) => {
            const riskColors = {
                critical: "text-error border-error bg-error/10",
                high: "text-amber-600 border-amber-500 bg-amber-500/10",
                medium: "text-blue-600 border-blue-500 bg-blue-500/10"
            };
            const cColor = riskColors[p.risk_level] || riskColors.medium;
            
            const hopsHTML = p.hops.map((h, i) => `
                <div class="relative pl-6 pb-4 border-l-2 border-slate-200 last:border-transparent last:pb-0">
                    <div class="absolute -left-[5px] top-1 w-2 h-2 rounded-full bg-primary-container z-10"></div>
                    <div class="bg-surface-container-low p-2 rounded-lg border border-outline-variant/30 text-[10px]">
                        <div class="flex justify-between font-bold text-primary-container mb-1">
                            <span>${h.from.substring(0,6)}... → ${h.to.substring(0,6)}...</span>
                            <span class="text-error">${h.amount_formatted}</span>
                        </div>
                        <div class="flex justify-between text-slate-500 font-mono">
                            <span>Ngày: ${h.date}</span>
                            <span>Prob: ${h.fraud_probability.toFixed(2)}</span>
                        </div>
                    </div>
                </div>
            `).join("");

            return `
                <div class="border border-outline-variant/30 rounded-xl p-3 mb-4 log-entry" style="animation: fadeSlideIn 0.4s ease ${pIdx * 0.1}s both;">
                    <div class="flex justify-between items-center mb-2">
                        <span class="text-xs font-black text-primary-container">${p.path_id}</span>
                        <span class="text-[9px] font-black uppercase px-2 py-0.5 rounded border ${cColor}">${p.risk_level}</span>
                    </div>
                    <p class="text-[10px] text-on-surface-variant font-medium leading-relaxed mb-3">${p.summary}</p>
                    <div class="mt-2">
                        ${hopsHTML}
                    </div>
                    <button class="w-full mt-3 py-1.5 flex items-center justify-center gap-1 text-[10px] uppercase font-bold text-primary-container bg-primary-container/5 hover:bg-primary-container/10 rounded transition-colors" onclick="highlightCycleNodes('${p.companies.join(',')}')">
                        <span class="material-symbols-outlined text-[14px]">visibility</span> Focus Chain
                    </button>
                </div>
            `;
        });
        
        if (data.evidence_paths.length > 0) {
            pathsContainer.innerHTML = pathHTML.join("");
        }
    }

    // Setup Tabs if not already done
    const tabLogs = document.getElementById("tab-logs");
    const tabPaths = document.getElementById("tab-paths");
    if (tabLogs && !tabLogs.hasAttribute('data-initialized')) {
        tabLogs.setAttribute('data-initialized', 'true');
        tabLogs.addEventListener("click", () => {
            tabLogs.className = "py-2 text-[10px] font-black uppercase text-primary-container border-b-2 border-primary-container tracking-widest transition-colors";
            tabPaths.className = "py-2 text-[10px] font-black uppercase text-slate-400 border-b-2 border-transparent hover:text-primary-container tracking-widest transition-colors";
            document.getElementById("investigation-logs").classList.replace("hidden", "block");
            document.getElementById("evidence-paths-container").classList.replace("block", "hidden");
        });
        tabPaths.addEventListener("click", () => {
            tabPaths.className = "py-2 text-[10px] font-black uppercase text-primary-container border-b-2 border-primary-container tracking-widest transition-colors flex items-center gap-1";
            tabLogs.className = "py-2 text-[10px] font-black uppercase text-slate-400 border-b-2 border-transparent hover:text-primary-container tracking-widest transition-colors";
            document.getElementById("investigation-logs").classList.replace("block", "hidden");
            document.getElementById("evidence-paths-container").classList.replace("hidden", "block");
        });
    }
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
        svg.selectAll(".edge").transition().duration(500).attr("opacity", d => d.is_circular ? 1 : 0.4);
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

    tooltip.innerHTML = `
        <p class="font-black text-sm mb-1">${d.label || d.tax_code}</p>
        <p class="text-slate-400 font-mono text-[10px] mb-2">${d.tax_code}</p>
        <div class="flex gap-4">
            <div><span class="text-slate-500">Ngành:</span> <span class="font-bold">${d.industry || 'N/A'}</span></div>
        </div>
        <div class="flex gap-4 mt-1">
            <div><span class="text-slate-500">Risk:</span> <span class="font-black" style="color:${riskColor}">${d.risk_score?.toFixed(0) || 0}%</span></div>
            <div><span class="text-slate-500">Loại:</span> <span class="font-bold">${d.is_shell ? '🔴 Shell Corp' : d.group === 'suspicious' ? '🟡 Nghi vấn' : '🟢 Bình thường'}</span></div>
        </div>
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

function showEmptyState() {
    const empty = document.getElementById("graph-empty-state");
    if (empty) empty.style.display = "flex";
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

function setupTimelineControls() {
    const playBtn = document.getElementById("timeline-play-btn");
    if (playBtn) {
        let isPlaying = false;
        playBtn.addEventListener("click", () => {
            isPlaying = !isPlaying;
            const icon = playBtn.querySelector("span");
            if (icon) icon.textContent = isPlaying ? "pause" : "play_arrow";
        });
    }
}

// ════════════════════════════════════════════════════════════════
//  Company List Integration
// ════════════════════════════════════════════════════════════════
async function loadCompanyList() {
    try {
        const res = await fetch(`${GRAPH_API_BASE}/graph/companies`, { credentials: "include" });
        if (!res.ok) throw new Error("Failed to load companies");
        const data = await res.json();
        allCompanies = data.results || [];
        filteredCompanies = [...allCompanies];
        renderCompanyTable();
    } catch (e) {
        console.error(e);
        const tbody = document.getElementById("companies-table-body");
        if (tbody) tbody.innerHTML = `<tr><td colspan="6" class="px-6 py-8 text-center text-error text-xs font-bold">Lỗi tải dữ liệu.</td></tr>`;
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
        const riskClass = c.risk_score >= 80 ? 'text-error border-error bg-error/10' :
                          c.risk_score >= 50 ? 'text-orange-600 border-orange-500 bg-orange-500/10' :
                          'text-emerald-600 border-emerald-500 bg-emerald-500/10';
        const riskLabel = c.risk_score >= 80 ? 'RẤT CAO' : c.risk_score >= 50 ? 'CAO' : 'BÌNH THƯỜNG';
        const statusBadge = c.is_active 
            ? `<span class="px-2 py-0.5 rounded border border-emerald-200 bg-emerald-50 text-emerald-700 text-[9px] font-bold uppercase">Hoạt động</span>`
            : `<span class="px-2 py-0.5 rounded border border-slate-200 bg-slate-50 text-slate-500 text-[9px] font-bold uppercase">Hủy/Ngừng</span>`;
        
        return `
            <tr class="hover:bg-slate-50 transition-colors">
                <td class="px-6 py-4 font-mono text-xs font-bold text-slate-600">${c.tax_code}</td>
                <td class="px-6 py-4 font-bold text-xs text-primary-container max-w-[200px] truncate" title="${c.name}">${c.name}</td>
                <td class="px-6 py-4 text-xs text-slate-500 truncate max-w-[150px]">${c.industry || 'N/A'}</td>
                <td class="px-6 py-4">${statusBadge}</td>
                <td class="px-6 py-4">
                    <div class="flex items-center gap-2">
                        <span class="text-xs font-black" style="color: ${c.risk_score >= 80 ? '#dc2626' : c.risk_score >= 50 ? '#ea580c' : '#16a34a'}">${c.risk_score.toFixed(0)}</span>
                        <span class="text-[8px] font-black uppercase px-1.5 py-0.5 rounded border ${riskClass}">${riskLabel}</span>
                    </div>
                </td>
                <td class="px-6 py-4 text-right">
                    <button onclick="analyzeCompanyGraph('${c.tax_code}')" class="px-3 py-1.5 bg-surface-container hover:bg-primary-container hover:text-white transition-colors rounded text-[10px] font-bold uppercase tracking-widest text-primary-container border border-outline-variant/30 flex items-center gap-1 inline-flex">
                        <span class="material-symbols-outlined text-[14px]">troubleshoot</span> Phân tích
                    </button>
                </td>
            </tr>
        `;
    }).join("");

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
