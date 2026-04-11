document.addEventListener("DOMContentLoaded", async () => {
    // Check if network container exists (added via python script)
    const container = document.getElementById('mynetwork');
    if (!container) return;

    try {
        const response = await fetch(`${API_BASE}/graph`);
        if (!response.ok) {
            throw new Error(`Graph API failed with status ${response.status}`);
        }
        const graphData = await response.json();
        const suspiciousNodeIds = new Set((graphData.suspicious_clusters || []).flat());

        // Map backend schema to Vis.js schema (supports both current and legacy fields).
        const nodes = new vis.DataSet(
            (graphData.nodes || []).map(n => {
                const nodeId = n.id ?? n.tax_code ?? n.name;
                const isSuspicious =
                    n.group === 'suspicious' ||
                    suspiciousNodeIds.has(nodeId) ||
                    Number(n.fraud_risk) > 0.7;
                const riskScore = Number(n.fraud_risk);
                const riskPercent = Number.isFinite(riskScore)
                    ? Math.round(riskScore * 100)
                    : (isSuspicious ? 90 : 20);

                return {
                    id: nodeId,
                    label: n.label ?? n.name ?? (nodeId ?? 'Unknown'),
                    title: `Risk: ${riskPercent}%`,
                    color: {
                        background: isSuspicious ? '#ba1a1a' : '#002147',
                        border: 'white'
                    },
                    font: { color: 'white', size: 10, face: 'Inter' },
                    shape: 'circle',
                    margin: 10
                };
            })
        );

        const edges = new vis.DataSet(
            (graphData.edges || [])
                .map(e => {
                    const fromNode = e.from_node ?? e.source;
                    const toNode = e.to_node ?? e.target;
                    const amount = Number(e.value ?? e.amount ?? 0);
                    const isSuspicious = Boolean(
                        e.is_suspicious ??
                        e.is_suspicous ??
                        (suspiciousNodeIds.has(fromNode) && suspiciousNodeIds.has(toNode))
                    );

                    return {
                        from: fromNode,
                        to: toNode,
                        label: amount > 0 ? `${(amount / 1000000).toFixed(0)}Tr` : '',
                        font: { align: 'top', size: 10, color: '#94a3b8' },
                        arrows: 'to',
                        color: { color: isSuspicious ? '#ba1a1a' : '#334155' },
                        dashes: isSuspicious // red-dashed circular path
                    };
                })
                .filter(e => e.from && e.to)
        );

        // Remove the SVG mock inside the container and initialize Vis.js
        container.innerHTML = '';
        
        const data = { nodes, edges };
        const options = {
            physics: {
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 100,
                    springConstant: 0.08
                }
            },
            interaction: { hover: true }
        };

        new vis.Network(container, data, options);

    } catch (error) {
        console.error("Graph Error:", error);
    }
});
