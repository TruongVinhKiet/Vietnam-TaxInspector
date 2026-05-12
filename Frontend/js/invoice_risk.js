
const API_BASE_INV = (typeof API_BASE !== "undefined" && API_BASE) || window.API_BASE_URL || "http://localhost:8000/api";

document.addEventListener('DOMContentLoaded', () => {
    const btnLookup = document.getElementById('inv-lookup-btn');
    const inputNumber = document.getElementById('inv-number-input');
    const btnHistory = document.getElementById('inv-list-load-btn');
    const inputMst = document.getElementById('inv-list-mst');
    const resultsArea = document.getElementById('analysis-results');

    let gaugeChart, radarChart, waterfallChart, controlChart, treemapChart;

    function initCharts() {
        gaugeChart = echarts.init(document.getElementById('chart-gauge'));
        radarChart = echarts.init(document.getElementById('chart-radar'));
        waterfallChart = echarts.init(document.getElementById('chart-waterfall'));
        controlChart = echarts.init(document.getElementById('chart-control'));
        treemapChart = echarts.init(document.getElementById('chart-treemap'));

        window.addEventListener('resize', () => {
            gaugeChart.resize(); radarChart.resize(); waterfallChart.resize();
            controlChart.resize(); treemapChart.resize();
        });
    }

    function renderGauge(score) {
        const option = {
            series: [{
                type: 'gauge',
                startAngle: 180, endAngle: 0,
                min: 0, max: 100,
                pointer: { show: true },
                progress: { show: true, overlap: false, roundCap: true, clip: false },
                axisLine: { lineStyle: { width: 14 } },
                splitLine: { show: false, distance: 0, length: 10 },
                axisTick: { show: false },
                axisLabel: { show: false, distance: 50 },
                data: [{ value: score, name: 'Điểm Rủi Ro', title: { offsetCenter: ['0%', '-30%'] }, detail: { offsetCenter: ['0%', '20%'] } }],
                title: { fontSize: 12, color: '#64748b' },
                detail: { width: 50, height: 14, fontSize: 24, color: 'inherit', formatter: '{value}%' }
            }]
        };
        gaugeChart.setOption(option);
    }

    function renderRadar(data) {
        const option = {
            radar: {
                indicator: data.indicators || [
                    { name: 'Thời gian xuất', max: 100 },
                    { name: 'Mạng lưới M/B', max: 100 },
                    { name: 'Giá bất thường', max: 100 },
                    { name: 'Tần suất', max: 100 },
                    { name: 'Mặt hàng', max: 100 }
                ],
                radius: 80
            },
            series: [{
                type: 'radar',
                data: [{ value: data.values || [80, 50, 90, 40, 60], name: 'Điểm thành phần' }],
                itemStyle: { color: '#e11d48' },
                areaStyle: { opacity: 0.2 }
            }]
        };
        radarChart.setOption(option);
    }

    function renderWaterfall(data) {
        const option = {
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
            xAxis: { type: 'category', data: data.categories || ['Base', 'Time penalty', 'Price penalty', 'Network', 'Total'] },
            yAxis: { type: 'value' },
            series: [
                { name: 'Placeholder', type: 'bar', stack: 'Total', itemStyle: { borderColor: 'transparent', color: 'transparent' }, emphasis: { itemStyle: { borderColor: 'transparent', color: 'transparent' } }, data: data.placeholders || [0, 20, 35, 65, 0] },
                { name: 'Penalty', type: 'bar', stack: 'Total', label: { show: true, position: 'top' }, data: data.values || [20, 15, 30, 20, 85] }
            ]
        };
        waterfallChart.setOption(option);
    }

    function renderControl(data) {
        const option = {
            tooltip: { trigger: 'axis' },
            xAxis: { type: 'category', data: data.dates },
            yAxis: { type: 'value', name: 'Risk Score' },
            series: [
                { name: 'Rủi ro', type: 'line', data: data.scores, markLine: { data: [{ type: 'average', name: 'Avg' }, { yAxis: 80, name: 'UCL' }] } }
            ]
        };
        controlChart.setOption(option);
    }

    function renderTreemap(data) {
        const option = {
            series: [{
                type: 'treemap',
                data: data.tree,
                label: { show: true, formatter: '{b}' },
                itemStyle: { borderColor: '#fff' }
            }]
        };
        treemapChart.setOption(option);
    }

    function renderTable(records) {
        const tbody = document.getElementById('inv-risk-table-body');
        if (!tbody || !records) return;
        tbody.innerHTML = records.map(r => `
            <tr>
                <td class="px-6 py-4 font-mono text-slate-500">${r.invoice_number}</td>
                <td class="px-6 py-4">${r.date}</td>
                <td class="px-6 py-4 font-bold text-primary-container">${r.buyer_name}</td>
                <td class="px-6 py-4 text-right font-mono">${r.amount}</td>
                <td class="px-6 py-4 text-center"><span class="px-2 py-1 bg-rose-100 text-rose-700 font-bold rounded">${r.risk_score}%</span></td>
                <td class="px-6 py-4 text-xs">${r.flags}</td>
            </tr>
        `).join('');
    }

    async function fetchData(taxCode) {
        resultsArea?.classList.remove('hidden');
        try {
            const fetchFn = typeof secureFetch === "function" ? secureFetch : fetch;
            const res = await fetchFn(`${API_BASE_INV}/invoice/analytics/${taxCode}`);
            const data = await res.json();

            if(data.gauge) renderGauge(data.gauge.score);
            if(data.radar) renderRadar(data.radar);
            if(data.waterfall) renderWaterfall(data.waterfall);
            if(data.control) renderControl(data.control);
            if(data.treemap) renderTreemap(data.treemap);
            if(data.records) renderTable(data.records);

        } catch (e) {
            console.error(e);
        }
    }

    let chartsInitialized = false;

    function safeInitCharts() {
        if (chartsInitialized) return true;
        const el = document.getElementById('chart-gauge');
        if (!el || el.offsetWidth === 0) return false;
        try {
            initCharts();
            chartsInitialized = true;
            return true;
        } catch(e) {
            console.warn('[InvRisk] Charts not ready yet');
            return false;
        }
    }

    btnLookup?.addEventListener('click', () => {
        if (safeInitCharts()) {
            fetchData("0108765432");
        }
    });

    btnHistory?.addEventListener('click', () => {
        const val = inputMst.value.trim() || '0108765432';
        if (safeInitCharts()) {
            fetchData(val);
        }
    });

    // Auto-load when NLP tab is clicked in fraud.html
    const nlpTabBtn = document.getElementById('tab-nlp-btn');
    let irAutoLoaded = false;
    if (nlpTabBtn) {
        nlpTabBtn.addEventListener('click', () => {
            setTimeout(() => {
                if (!irAutoLoaded && safeInitCharts()) {
                    irAutoLoaded = true;
                    fetchData("0108765432");
                }
            }, 300); // small delay for tab animation
        });
    }
});
