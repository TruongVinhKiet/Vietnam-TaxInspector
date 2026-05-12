
const API_BASE_TP = (typeof API_BASE !== "undefined" && API_BASE) || window.API_BASE_URL || "http://localhost:8000/api";

document.addEventListener('DOMContentLoaded', () => {
    const btnRunScore = document.getElementById('tp-run-score');
    const tableBody = document.getElementById('tp-table-body');
    const tableStatus = document.getElementById('tp-table-status');
    const tableLoading = document.getElementById('tp-table-loading');

    let scatterChart = null;
    let sankeyChart = null;
    let boxplotChart = null;
    let divergingChart = null;

    function initCharts() {
        scatterChart = echarts.init(document.getElementById('chart-scatter'));
        sankeyChart = echarts.init(document.getElementById('chart-sankey'));
        boxplotChart = echarts.init(document.getElementById('chart-boxplot'));
        divergingChart = echarts.init(document.getElementById('chart-diverging'));

        window.addEventListener('resize', () => {
            scatterChart.resize();
            sankeyChart.resize();
            boxplotChart.resize();
            divergingChart.resize();
        });
    }

    function renderScatterChart(data) {
        const option = {
            tooltip: {
                trigger: 'item',
                formatter: function(params) {
                    return `<div class="text-xs font-sans">
                        <b>Trạng thái:</b> ${params.value[3]}<br/>
                        <b>Khối lượng:</b> ${params.value[0]}<br/>
                        <b>Đơn giá:</b> ${params.value[1].toLocaleString()} đ<br/>
                        <b>Z-Score:</b> ${params.value[2].toFixed(2)}
                    </div>`;
                }
            },
            xAxis: { type: 'value', name: 'Khối lượng' },
            yAxis: { type: 'value', name: 'Đơn giá (VND)' },
            series: [{
                type: 'scatter',
                data: data,
                symbolSize: function (val) {
                    return Math.min(Math.max(val[2] * 4, 8), 40);
                },
                itemStyle: {
                    color: function(params) {
                        return params.value[2] > 2 ? '#e11d48' : '#3b82f6';
                    },
                    opacity: 0.7
                }
            }]
        };
        scatterChart.setOption(option);
    }

    function renderSankeyChart(data) {
        if (!data || !data.nodes || !data.links) return;
        const option = {
            tooltip: { trigger: 'item', triggerOn: 'mousemove' },
            series: {
                type: 'sankey',
                layout: 'none',
                data: data.nodes,
                links: data.links,
                lineStyle: { color: 'source', curveness: 0.5, opacity: 0.4 },
                label: { fontFamily: 'Inter', fontSize: 10, color: '#475569' }
            }
        };
        sankeyChart.setOption(option);
    }

    function renderBoxplotChart(data) {
        if (!data || !data.box_data) return;
        const option = {
            tooltip: { trigger: 'item' },
            xAxis: { type: 'category', data: data.categories },
            yAxis: { type: 'value', name: 'Đơn giá' },
            series: [
                {
                    name: 'Phân phối giá',
                    type: 'boxplot',
                    itemStyle: { color: '#0ea5e9', borderColor: '#0369a1' },
                    data: data.box_data
                },
                {
                    name: 'Ngoại lai',
                    type: 'scatter',
                    symbolSize: 8,
                    itemStyle: { color: '#e11d48' },
                    data: data.outliers
                }
            ]
        };
        boxplotChart.setOption(option);
    }

    function renderDivergingChart(data) {
        if (!data) return;
        const option = {
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            grid: { top: 30, bottom: 30, left: 100, right: 30 },
            xAxis: { type: 'value', position: 'top', name: 'Độ lệch %', axisLabel: { formatter: '{value} %' } },
            yAxis: { type: 'category', data: data.categories, axisLine: { show: false }, axisTick: { show: false } },
            series: [{
                name: 'Độ lệch giá',
                type: 'bar',
                barWidth: 20,
                label: { show: true, position: 'inside', formatter: '{b}', color: '#fff', fontSize: 10 },
                itemStyle: {
                    borderRadius: 4,
                    color: function(params) { return params.value < 0 ? '#e11d48' : '#0ea5e9'; }
                },
                data: data.values
            }]
        };
        divergingChart.setOption(option);
    }

    function renderTable(records) {
        if (!records) return;
        tableBody.innerHTML = records.map(r => {
            const absZ = Math.abs(r.zscore);
            const barWidth = Math.min(absZ * 15, 100);
            const barColor = r.zscore < 0 ? 'bg-rose-500' : 'bg-sky-500';
            const riskColor = r.risk > 90 ? 'bg-red-100 text-red-700' : (r.risk > 70 ? 'bg-orange-100 text-orange-700' : 'bg-yellow-100 text-yellow-700');

            return `
                <tr class="hover:bg-slate-50 transition-colors">
                    <td class="px-6 py-4 text-xs font-mono text-slate-500">${r.id}</td>
                    <td class="px-6 py-4 text-sm font-semibold text-primary-container">${r.mst}</td>
                    <td class="px-6 py-4 text-xs font-medium text-slate-700">${r.item}</td>
                    <td class="px-6 py-4 text-sm text-right font-mono text-slate-600">${r.price}</td>
                    <td class="px-6 py-4">
                        <div class="flex items-center gap-2">
                            <span class="text-xs font-mono w-8 text-right ${r.zscore < 0 ? 'text-rose-600' : 'text-sky-600'}">${r.zscore > 0 ? '+'+r.zscore : r.zscore}</span>
                            <div class="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden flex ${r.zscore < 0 ? 'justify-end' : 'justify-start'}">
                                <div class="h-full rounded-full ${barColor}" style="width: ${barWidth}%"></div>
                            </div>
                        </div>
                    </td>
                    <td class="px-6 py-4 text-center">
                        <span class="px-2.5 py-1 rounded-md text-[11px] font-black ${riskColor}">${r.risk}</span>
                    </td>
                </tr>
            `;
        }).join('');
    }

    async function loadTPData() {
        try {
            const fetchFn = typeof secureFetch === "function" ? secureFetch : fetch;
            const res = await fetchFn(`${API_BASE_TP}/transfer-pricing/analytics`);
            const data = await res.json();

            if(data.summary) {
                const el = (id) => document.getElementById(id);
                if(el('kpi-total-records')) el('kpi-total-records').innerText = data.summary.total_records.toLocaleString();
                if(el('kpi-anomalies')) el('kpi-anomalies').innerText = data.summary.anomalies.toLocaleString();
                if(el('kpi-avg-zscore')) el('kpi-avg-zscore').innerText = data.summary.avg_zscore.toFixed(2);
                if(el('kpi-risk-value')) el('kpi-risk-value').innerText = data.summary.risk_value;
            }

            if(data.scatter) renderScatterChart(data.scatter);
            if(data.sankey) renderSankeyChart(data.sankey);
            if(data.boxplot) renderBoxplotChart(data.boxplot);
            if(data.diverging) renderDivergingChart(data.diverging);
            if(data.records) renderTable(data.records);

            if(tableStatus) {
                tableStatus.innerHTML = `Quét thành công ${data.summary?.total_records || 0} bản ghi.`;
            }
        } catch(e) {
            console.error('[TP Analytics]', e);
            if(tableStatus) tableStatus.innerHTML = `<span class="text-error">Lỗi khi tải dữ liệu Transfer Pricing.</span>`;
        }
    }

    // Button handler (if exists on standalone page)
    btnRunScore?.addEventListener('click', loadTPData);

    // Auto-load when TP tab becomes visible (inside graph.html)
    let tpLoaded = false;
    const tpSection = document.getElementById('tp-section');
    if (tpSection) {
        const observer = new MutationObserver(() => {
            if (!tpLoaded && tpSection.style.display !== 'none' && !tpSection.classList.contains('hidden')) {
                tpLoaded = true;
                // Wait for DOM layout to complete before initializing ECharts to prevent width=0 issues
                setTimeout(() => {
                    initCharts();
                    loadTPData();
                }, 100);
            }
        });
        observer.observe(tpSection, { attributes: true, attributeFilter: ['style', 'class'] });
    }

});
