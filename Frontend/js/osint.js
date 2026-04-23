/**
 * osint.js – OSINT & UBO Intelligence Frontend Logic
 * ====================================================
 * Handles UBO graph visualization (D3-free canvas rendering),
 * high-risk entity listing, search, and jurisdiction stats.
 */

const OSINT_API = `${API_BASE}/osint`;

// ── State ──────────────────────────────────────────────────
let osintGraphData = null;
let currentPage = 1;
const PAGE_SIZE = 20;

// ── DOM Ready ──────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
    await checkAuth();
    await Promise.all([loadStats(), loadJurisdictions(), loadHighRiskUBO()]);
    bindOsintEvents();
});

// ── Load Stats ─────────────────────────────────────────────
async function loadStats() {
    try {
        const res = await secureFetch(`${OSINT_API}/stats`);
        if (!res.ok) return;
        const stats = await res.json();

        setText("osint-total-nodes", stats.total_nodes.toLocaleString());
        setText("osint-total-edges", stats.total_edges.toLocaleString());
        setText("osint-offshore-count", stats.total_offshore_entities.toLocaleString());
        setText("osint-high-risk-count", stats.high_risk_count.toLocaleString());
        setText("osint-avg-risk", stats.avg_risk_score.toFixed(1));
    } catch (e) {
        console.error("[OSINT] Stats load failed:", e);
    }
}

// ── Load Jurisdictions ─────────────────────────────────────
async function loadJurisdictions() {
    try {
        const res = await secureFetch(`${OSINT_API}/countries`);
        if (!res.ok) return;
        const data = await res.json();
        const container = document.getElementById("osint-jurisdictions");
        if (!container) return;

        container.innerHTML = data.map(j => {
            const riskColor = j.avg_risk_score >= 80 ? "bg-rose-500" : j.avg_risk_score >= 60 ? "bg-amber-500" : "bg-emerald-500";
            return `
                <div class="flex items-center justify-between px-4 py-3 border-b border-slate-100 hover:bg-slate-50 transition-colors cursor-pointer" data-country="${j.country}">
                    <div class="flex items-center gap-3">
                        <div class="w-2.5 h-2.5 rounded-full ${riskColor}"></div>
                        <span class="text-xs font-bold text-slate-700">${j.country}</span>
                    </div>
                    <div class="flex items-center gap-4 text-[10px]">
                        <span class="text-slate-500">${j.entity_count} entities</span>
                        <span class="font-black text-slate-700">${j.avg_risk_score.toFixed(1)} risk</span>
                        <span class="text-slate-400">${j.total_connections} links</span>
                    </div>
                </div>`;
        }).join("");
    } catch (e) {
        console.error("[OSINT] Jurisdictions load failed:", e);
    }
}

// ── Load High Risk UBO ─────────────────────────────────────
async function loadHighRiskUBO(page = 1) {
    currentPage = page;
    try {
        const res = await secureFetch(`${OSINT_API}/high-risk-ubo?page=${page}&page_size=${PAGE_SIZE}&min_risk=50`);
        if (!res.ok) return;
        const data = await res.json();

        const tbody = document.getElementById("osint-ubo-tbody");
        if (!tbody) return;

        if (!data.items.length) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-xs text-slate-400 py-8">Không tìm thấy dữ liệu.</td></tr>';
            return;
        }

        tbody.innerHTML = data.items.map((item, index) => {
            const riskColor = item.risk_score >= 80 ? "text-rose-600 bg-rose-50" : item.risk_score >= 60 ? "text-amber-600 bg-amber-50" : "text-emerald-600 bg-emerald-50";
            const relBadges = item.relation_types.map(r => {
                const colors = {
                    director_overlap: "bg-violet-50 text-violet-700 border-violet-200",
                    shares_ownership: "bg-sky-50 text-sky-700 border-sky-200",
                    suspicious_wire_transfer: "bg-rose-50 text-rose-700 border-rose-200",
                    Owner: "bg-sky-50 text-sky-700 border-sky-200",
                    Subsidiary: "bg-emerald-50 text-emerald-700 border-emerald-200",
                    RelatedParty: "bg-amber-50 text-amber-700 border-amber-200",
                    Relative: "bg-violet-50 text-violet-700 border-violet-200",
                };
                const labels = {
                    director_overlap: "Giám đốc trùng",
                    shares_ownership: "Sở hữu cổ phần",
                    suspicious_wire_transfer: "Chuyển tiền nghi vấn",
                    Owner: "Chủ sở hữu",
                    Subsidiary: "Công ty con",
                    RelatedParty: "Bên liên quan",
                    Relative: "Họ hàng",
                };
                return `<span class="px-1.5 py-0.5 rounded text-[9px] font-bold border ${colors[r] || 'bg-slate-50 text-slate-600 border-slate-200'}">${labels[r] || r}</span>`;
            }).join(" ");

            const delay = index * 0.05;
            return `
                <tr class="border-b border-slate-100 hover:bg-slate-50 transition-colors page-transition" style="animation-delay: ${delay}s">
                    <td class="px-4 py-3">
                        <div class="text-xs font-black text-primary-container">${item.label}</div>
                        <div class="text-[10px] text-slate-400 mt-0.5">${item.offshore_id}</div>
                    </td>
                    <td class="px-4 py-3 text-xs font-bold text-slate-600">${item.country}</td>
                    <td class="px-4 py-3"><span class="px-2 py-1 rounded-lg text-xs font-black ${riskColor}">${item.risk_score.toFixed(1)}</span></td>
                    <td class="px-4 py-3 text-xs text-slate-600">${item.connected_domestic_count}</td>
                    <td class="px-4 py-3"><div class="flex flex-wrap gap-1">${relBadges}</div></td>
                    <td class="px-4 py-3">
                        <button class="osint-view-graph-btn px-3 py-1.5 rounded-lg bg-primary-container text-white text-[10px] font-bold hover:opacity-90 transition-opacity"
                                data-tax-codes="${item.top_domestic_tax_codes.join(',')}"
                                data-proxy-tax-code="${item.proxy_tax_code || ''}">
                            <span class="material-symbols-outlined text-[14px]">hub</span> Xem mạng
                        </button>
                    </td>
                </tr>`;
        }).join("");

        // Pagination
        const pagInfo = document.getElementById("osint-pag-info");
        if (pagInfo) pagInfo.textContent = `Trang ${data.page}/${Math.ceil(data.total / data.page_size)} (${data.total} kết quả)`;

        const pagContainer = document.getElementById("osint-pagination");
        if (pagContainer) {
            const totalPages = Math.ceil(data.total / data.page_size);
            let html = "";
            if (page > 1) html += `<button class="osint-page-btn px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-bold" data-page="${page - 1}">‹ Trước</button>`;
            for (let p = Math.max(1, page - 2); p <= Math.min(totalPages, page + 2); p++) {
                const active = p === page ? "bg-primary-container text-white" : "bg-white text-slate-600 border border-slate-200";
                html += `<button class="osint-page-btn px-3 py-1.5 rounded-lg text-xs font-bold ${active}" data-page="${p}">${p}</button>`;
            }
            if (page < totalPages) html += `<button class="osint-page-btn px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-bold" data-page="${page + 1}">Sau ›</button>`;
            pagContainer.innerHTML = html;
            pagContainer.querySelectorAll(".osint-page-btn").forEach(b => b.addEventListener("click", () => loadHighRiskUBO(parseInt(b.dataset.page))));
        }

        // Bind view graph buttons
        tbody.querySelectorAll(".osint-view-graph-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const codes = btn.dataset.taxCodes?.split(",").filter(Boolean) || [];
                const proxyTaxCode = String(btn.dataset.proxyTaxCode || "").trim();
                const targetTaxCode = proxyTaxCode || (codes.length ? codes[0] : "");
                if (!targetTaxCode) return;
                const osintTab = document.getElementById("workbench-tab-osint");
                if (osintTab) osintTab.click();
                loadOsintGraph(targetTaxCode);
            });
        });
    } catch (e) {
        console.error("[OSINT] UBO list load failed:", e);
    }
}

// ── Load & Render Graph ────────────────────────────────────
async function loadOsintGraph(taxCode) {
    const container = document.getElementById("osint-graph-canvas");
    const statusEl = document.getElementById("osint-graph-status");
    if (!container) return;

    if (statusEl) statusEl.innerHTML = `<span class="material-symbols-outlined text-[14px] animate-spin">autorenew</span> Đang tải mạng lưới MST ${taxCode}...`;

    try {
        const res = await secureFetch(`${OSINT_API}/graph/${taxCode}?depth=2`);
        if (!res.ok) {
            if (statusEl) statusEl.innerHTML = `<span class="text-rose-500">Không tìm thấy dữ liệu OSINT cho MST ${taxCode}</span>`;
            return;
        }
        osintGraphData = await res.json();
        if (statusEl) statusEl.innerHTML = `MST <strong>${taxCode}</strong> — ${osintGraphData.total_connections} liên kết — ${osintGraphData.offshore_jurisdictions.length} quốc gia`;
        renderOsintGraphCanvas(container, osintGraphData);

        // Scroll to graph section
        container.scrollIntoView({ behavior: "smooth", block: "center" });
    } catch (e) {
        console.error("[OSINT] Graph load failed:", e);
        if (statusEl) statusEl.innerHTML = `<span class="text-rose-500">Lỗi tải graph: ${e.message}</span>`;
    }
}

function renderOsintGraphCanvas(container, data) {
    const canvas = container.querySelector("canvas") || document.createElement("canvas");
    if (!canvas.parentNode) container.appendChild(canvas);

    const dpr = window.devicePixelRatio || 1;
    const W = container.clientWidth;
    const H = 500;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const nodes = data.nodes || [];
    const edges = data.edges || [];
    if (!nodes.length) return;

    // Simple force-directed layout (spring simulation)
    const positions = {};
    const centerX = W / 2;
    const centerY = H / 2;

    nodes.forEach((n, i) => {
        const angle = (i / nodes.length) * Math.PI * 2;
        const radius = n.type === "domestic_entity" ? 80 : 150 + Math.random() * 80;
        positions[n.id] = {
            x: centerX + Math.cos(angle) * radius,
            y: centerY + Math.sin(angle) * radius,
            vx: 0,
            vy: 0,
        };
    });

    // Place center node at center
    if (data.center_node) {
        positions[data.center_node.id] = { x: centerX, y: centerY, vx: 0, vy: 0 };
    }

    // Animated drawing loop
    let iter = 0;
    
    function animate() {
        if (iter >= 80) return; // Limit iterations so it stops computing
        iter++;
        
        ctx.clearRect(0, 0, W, H);

        // Repulsion between all nodes
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const a = positions[nodes[i].id];
                const b = positions[nodes[j].id];
                const dx = b.x - a.x;
                const dy = b.y - a.y;
                const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
                const force = 800 / (dist * dist);
                a.vx -= (dx / dist) * force;
                a.vy -= (dy / dist) * force;
                b.vx += (dx / dist) * force;
                b.vy += (dy / dist) * force;
            }
        }

        // Attraction along edges
        edges.forEach(e => {
            const a = positions[e.source];
            const b = positions[e.target];
            if (!a || !b) return;
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
            const force = (dist - 120) * 0.01;
            a.vx += (dx / dist) * force;
            a.vy += (dy / dist) * force;
            b.vx -= (dx / dist) * force;
            b.vy -= (dy / dist) * force;
        });

        // Apply velocities with damping
        nodes.forEach(n => {
            const p = positions[n.id];
            p.x += p.vx * 0.5;
            p.y += p.vy * 0.5;
            p.vx *= 0.8;
            p.vy *= 0.8;
            // Keep in bounds
            p.x = Math.max(60, Math.min(W - 60, p.x));
            p.y = Math.max(60, Math.min(H - 60, p.y));
        });

        // Pin center node
        if (data.center_node && positions[data.center_node.id]) {
            positions[data.center_node.id].x = centerX;
            positions[data.center_node.id].y = centerY;
        }

        // Draw edges
        edges.forEach(e => {
            const a = positions[e.source];
            const b = positions[e.target];
            if (!a || !b) return;

            const colors = {
                director_overlap: "#8b5cf6",
                shares_ownership: "#0284c7",
                suspicious_wire_transfer: "#ef4444",
            };

            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = colors[e.relation_type] || "#cbd5e1";
            ctx.lineWidth = 1 + (e.weight || 0.5);
            ctx.globalAlpha = 0.4;
            ctx.stroke();
            ctx.globalAlpha = 1;
        });

        // Draw nodes
        nodes.forEach(n => {
            const p = positions[n.id];
            const isCenter = data.center_node && n.id === data.center_node.id;
            const isOffshore = n.type === "offshore_entity";
            const r = isCenter ? 16 : isOffshore ? 10 : 7;

            // Glow for center
            if (isCenter) {
                ctx.beginPath();
                ctx.arc(p.x, p.y, r + 8, 0, Math.PI * 2);
                ctx.fillStyle = "rgba(2, 132, 199, 0.15)";
                ctx.fill();
            }

            ctx.beginPath();
            ctx.arc(p.x, p.y, r, 0, Math.PI * 2);

            if (isCenter) {
                ctx.fillStyle = "#0284c7";
            } else if (isOffshore) {
                const risk = n.risk_score || 50;
                ctx.fillStyle = risk >= 80 ? "#ef4444" : risk >= 60 ? "#f59e0b" : "#10b981";
            } else {
                ctx.fillStyle = "#64748b";
            }
            ctx.fill();
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 2;
            ctx.stroke();

            // Label
            if (isCenter || isOffshore) {
                ctx.font = `${isCenter ? "bold 11px" : "10px"} Inter, sans-serif`;
                ctx.fillStyle = "#334155";
                ctx.textAlign = "center";
                const label = n.label.length > 20 ? n.label.slice(0, 18) + "..." : n.label;
                ctx.fillText(label, p.x, p.y + r + 14);
                if (isOffshore && n.country) {
                    ctx.font = "9px Inter, sans-serif";
                    ctx.fillStyle = "#94a3b8";
                    ctx.fillText(n.country, p.x, p.y + r + 26);
                }
            }
        });

        if (iter < 80) requestAnimationFrame(animate);
    }
    
    // Start animation loop
    animate();
}

// ── Search ─────────────────────────────────────────────────
async function searchOsint(query) {
    if (!query || query.length < 2) return;
    try {
        const res = await secureFetch(`${OSINT_API}/search?q=${encodeURIComponent(query)}&limit=10`);
        if (!res.ok) return;
        const results = await res.json();
        const dropdown = document.getElementById("osint-search-results");
        if (!dropdown) return;

        if (!results.length) {
            dropdown.innerHTML = '<div class="px-4 py-3 text-xs text-slate-400">Không tìm thấy kết quả.</div>';
            dropdown.classList.remove("hidden");
            return;
        }

        dropdown.innerHTML = results.map(r => {
            const icon = r.node.type === "offshore_entity" ? "public" : "business";
            const color = r.node.type === "offshore_entity" ? "text-rose-500" : "text-sky-500";
            return `
                <button class="osint-search-item w-full text-left px-4 py-2.5 hover:bg-slate-50 transition-colors flex items-center gap-3"
                        data-id="${r.node.id}" data-type="${r.node.type}" data-tax-code="${r.node.tax_code || ''}">
                    <span class="material-symbols-outlined ${color} text-[18px]">${icon}</span>
                    <div>
                        <div class="text-xs font-bold text-slate-700">${r.node.label}</div>
                        <div class="text-[10px] text-slate-400">${r.node.country || 'Việt Nam'} · ${r.connections} liên kết · ${r.match_type}</div>
                    </div>
                </button>`;
        }).join("");
        dropdown.classList.remove("hidden");

        dropdown.querySelectorAll(".osint-search-item").forEach(item => {
            item.addEventListener("click", () => {
                dropdown.classList.add("hidden");
                const taxCode = item.dataset.taxCode;
                const nodeId = item.dataset.id;
                if (taxCode) {
                    loadOsintGraph(taxCode);
                } else if (nodeId.startsWith("DOM_")) {
                    loadOsintGraph(nodeId.replace("DOM_", ""));
                }
            });
        });
    } catch (e) {
        console.error("[OSINT] Search failed:", e);
    }
}

// ── Events ─────────────────────────────────────────────────
function bindOsintEvents() {
    const searchInput = document.getElementById("osint-search-input");
    let debounce = null;
    if (searchInput) {
        searchInput.addEventListener("input", () => {
            clearTimeout(debounce);
            debounce = setTimeout(() => searchOsint(searchInput.value.trim()), 400);
        });
        searchInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                searchOsint(searchInput.value.trim());
            }
        });
        // Close dropdown on click outside
        document.addEventListener("click", (e) => {
            const dropdown = document.getElementById("osint-search-results");
            if (dropdown && !searchInput.contains(e.target) && !dropdown.contains(e.target)) {
                dropdown.classList.add("hidden");
            }
        });
    }

    // Direct MST lookup
    const lookupBtn = document.getElementById("osint-lookup-btn");
    const lookupInput = document.getElementById("osint-lookup-input");
    if (lookupBtn && lookupInput) {
        lookupBtn.addEventListener("click", () => {
            const code = lookupInput.value.trim();
            if (code) loadOsintGraph(code);
        });
        lookupInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                const code = lookupInput.value.trim();
                if (code) loadOsintGraph(code);
            }
        });
    }
}

// ── Helper ─────────────────────────────────────────────────
function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}
