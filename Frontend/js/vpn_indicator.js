// vpn_indicator.js – VPN Forensic Badge Overlay for D3.js Graph Nodes
// Renders animated warning badges on nodes flagged with VPN evasion indicators.

/**
 * Show VPN forensic detail in a styled modal panel instead of alert().
 */
function showVPNDetail(nodeData) {
    const score = (nodeData.vpn_score || 0).toFixed(2);
    const taxCode = nodeData.id || nodeData.tax_code || '—';
    const layers = nodeData.vpn_layers || {};

    // Build layer breakdown HTML
    const layerRows = Object.entries(layers).map(([k, v]) =>
        `<tr><td class="pr-3 text-slate-400 text-xs">${k}</td><td class="font-mono text-xs">${Number(v).toFixed(3)}</td></tr>`
    ).join('') || '<tr><td colspan="2" class="text-slate-500 text-xs">Không có dữ liệu chi tiết</td></tr>';

    // Create or reuse modal
    let modal = document.getElementById('vpn-detail-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'vpn-detail-modal';
        modal.className = 'fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm';
        document.body.appendChild(modal);
    }

    modal.innerHTML = `
        <div class="bg-slate-900 border border-red-500/30 rounded-xl shadow-2xl max-w-md w-full mx-4 p-6 relative animate-fade-in">
            <button onclick="document.getElementById('vpn-detail-modal').remove()"
                class="absolute top-3 right-3 text-slate-400 hover:text-white text-xl leading-none">&times;</button>
            <div class="flex items-center gap-3 mb-4">
                <div class="w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center text-lg">🛡️</div>
                <div>
                    <div class="text-white font-bold text-sm">Phân Tích Pháp Y Mạng</div>
                    <div class="text-slate-400 text-xs font-mono">MST: ${taxCode}</div>
                </div>
            </div>
            <div class="grid grid-cols-2 gap-3 mb-4">
                <div class="bg-slate-800 rounded-lg p-3 text-center">
                    <div class="text-xs text-slate-400 mb-1">Composite Score</div>
                    <div class="text-2xl font-black ${score >= 0.7 ? 'text-red-400' : score >= 0.55 ? 'text-amber-400' : 'text-green-400'}">${score}</div>
                </div>
                <div class="bg-slate-800 rounded-lg p-3 text-center">
                    <div class="text-xs text-slate-400 mb-1">Verdict</div>
                    <div class="text-sm font-bold ${score >= 0.55 ? 'text-red-400' : 'text-green-400'}">${score >= 0.55 ? '⚠️ VPN Detected' : '✅ Clean'}</div>
                </div>
            </div>
            <div class="bg-slate-800/50 rounded-lg p-3">
                <div class="text-xs text-slate-400 mb-2 font-semibold">Phân tích theo tầng (5-Layer)</div>
                <table class="w-full">${layerRows}</table>
            </div>
        </div>
    `;
    modal.style.display = 'flex';

    // Close on backdrop click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
}

/**
 * Render VPN forensic badges on D3 node groups.
 * Called from graph.js after node labels are created.
 */
window.renderVPNBadge = function(d3NodeGroupSelection) {
    d3NodeGroupSelection.each(function(nodeData) {
        if (nodeData.vpn_flag || nodeData.vpn_score > 0.55) {
            const group = d3.select(this);

            // Prevent duplicate badges on re-render
            if (!group.select(".vpn-badge-group").empty()) return;

            const badgeGroup = group.append("g")
                .attr("class", "vpn-badge-group")
                .attr("transform", "translate(40, -16)")
                .style("cursor", "pointer")
                .on("click", (event, d) => {
                    event.stopPropagation();
                    showVPNDetail(nodeData);
                });

            // Outer pulse ring (LARGER than inner – CSS animated)
            badgeGroup.append("circle")
                .attr("r", 12)
                .attr("fill", "none")
                .attr("stroke", "#ef4444")
                .attr("stroke-width", 2)
                .attr("opacity", 0.6)
                .attr("class", "vpn-pulse-ring");

            // Inner solid badge
            badgeGroup.append("circle")
                .attr("r", 7)
                .attr("fill", "#ef4444")
                .attr("stroke", "#ffffff")
                .attr("stroke-width", 1.5);

            // Shield icon
            badgeGroup.append("text")
                .attr("text-anchor", "middle")
                .attr("dy", 3)
                .attr("font-size", "9px")
                .attr("fill", "#ffffff")
                .attr("font-weight", "bold")
                .text("V");
        }
    });
};
