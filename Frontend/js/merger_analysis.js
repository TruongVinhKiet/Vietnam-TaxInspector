// merger_analysis.js - pre/post provincial merger economic analysis panel.

const MergerAnalysis = (() => {
    let chart = null;
    let latestSelection = null;

    function byId(id) {
        return document.getElementById(id);
    }

    function esc(value) {
        const helper = window.MacroMapData?.escapeHtml;
        if (helper) return helper(value);
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    async function showForProvince(provinceCode, provinceName = '') {
        const panel = byId('merger-analysis-panel');
        if (!panel || !provinceCode || !window.MergerMapData) return;
        latestSelection = { provinceCode, provinceName };
        panel.classList.remove('hidden');
        // Removed auto-scroll: let users scroll manually

        const title = byId('merger-analysis-title');
        const status = byId('merger-analysis-status');
        if (title) title.textContent = `Phân tích kinh tế sáp nhập - ${provinceName || provinceCode}`;
        if (status) status.textContent = 'Đang tải chuỗi GRDP, sự kiện và dữ liệu dự phóng...';

        try {
            const data = await window.MergerMapData.getMergerAnalysis(provinceCode, {
                boundaryVersion: window.MACRO_BOUNDARY_VERSION || 'vn_34_2025',
            });
            if (latestSelection?.provinceCode !== provinceCode) return;
            renderSummary(data);
            renderChart(data);
            renderSectorChart(data);
            renderTable(data);
            renderEventCards(data);
            // Dispatch event for enhancement charts (Radar, Sector Stacked Bar)
            window.dispatchEvent(new CustomEvent('merger:data-loaded', { detail: data }));
            if (status) {
                const quality = data.source_quality || {};
                status.textContent = `${quality.observed_level || 'national_observed_province_estimated'} - ${quality.data_window || '2019-2025'}`;
            }
        } catch (error) {
            if (status) status.textContent = `Không tải được phân tích sáp nhập: ${error.message || error}`;
        }
    }

    function renderSummary(data) {
        const container = byId('merger-summary-cards');
        if (!container) return;
        const unit = data.new_unit || {};
        const growth = data.merged_growth || {};
        const post = data.post_merger_baseline || {};
        const members = Array.isArray(data.member_rows) ? data.member_rows : [];
        container.innerHTML = `
            <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div class="text-[10px] font-black uppercase tracking-wider text-slate-500">Đơn vị sau sáp nhập</div>
                <div class="mt-1 text-lg font-black text-primary-container">${esc(unit.province_name || '--')}</div>
                <div class="mt-1 text-xs text-slate-500">${members.length} tỉnh/thành thành viên</div>
            </div>
            <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div class="text-[10px] font-black uppercase tracking-wider text-slate-500">Tăng trưởng 2019-2024</div>
                <div class="mt-1 text-lg font-black ${Number(growth.growth_pct || 0) >= 0 ? 'text-emerald-600' : 'text-red-600'}">${window.MergerMapData.formatPercent(growth.growth_pct)}</div>
                <div class="mt-1 text-xs text-slate-500">CAGR ${window.MergerMapData.formatPercent(growth.cagr_pct)}</div>
            </div>
            <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div class="text-[10px] font-black uppercase tracking-wider text-slate-500">Mốc sau sáp nhập</div>
                <div class="mt-1 text-lg font-black ${Number(post.delta_pct_est || 0) >= 0 ? 'text-emerald-600' : 'text-amber-600'}">${window.MergerMapData.formatPercent(post.delta_pct_est)}</div>
                <div class="mt-1 text-xs text-slate-500">2025 là ước lượng có provenance</div>
            </div>
            <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div class="text-[10px] font-black uppercase tracking-wider text-slate-500">Trung tâm hành chính</div>
                <div class="mt-1 text-lg font-black text-primary-container">${esc(unit.political_admin_center || '--')}</div>
                <div class="mt-1 text-xs text-slate-500">Theo bảng sáp nhập đã duyệt</div>
            </div>
        `;
    }

    function renderChart(data) {
        const canvas = byId('merger-grdp-chart');
        if (!canvas || typeof Chart === 'undefined') return;
        const merged = (data.merged_time_series || []).filter((row) => Number(row.year) >= 2019);
        const labels = merged.map((row) => String(row.year));
        const datasets = [{
            label: `${data.new_unit?.province_name || 'Đơn vị mới'} - tổng hợp`,
            data: merged.map((row) => Number(row.grdp_billion_vnd_est || 0)),
            borderColor: '#002147',
            backgroundColor: 'rgba(0,33,71,0.08)',
            borderWidth: 3,
            tension: 0.25,
            fill: true,
        }];
        const palette = ['#0ea5e9', '#22c55e', '#f97316', '#a855f7'];
        (data.member_rows || []).slice(0, 4).forEach((member, idx) => {
            const rows = (member.time_series || []).filter((row) => Number(row.year) >= 2019);
            datasets.push({
                label: member.province_name || member.province_code,
                data: labels.map((year) => {
                    const row = rows.find((item) => String(item.year) === String(year));
                    return row ? Number(row.grdp_billion_vnd_est || 0) : null;
                }),
                borderColor: palette[idx % palette.length],
                borderWidth: 1.8,
                tension: 0.25,
                fill: false,
                pointRadius: 2,
            });
        });
        if (chart) chart.destroy();
        chart = new Chart(canvas, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${Number(ctx.parsed.y || 0).toLocaleString('vi-VN')} tỷ`,
                        },
                    },
                },
                scales: {
                    y: {
                        ticks: { callback: (value) => `${Number(value).toLocaleString('vi-VN')} tỷ` },
                        grid: { color: 'rgba(148,163,184,0.2)' },
                    },
                    x: { grid: { display: false } },
                },
            },
        });
    }

    function renderSectorChart(data) {
        const canvas = byId('merger-sector-chart');
        if (!canvas || typeof Chart === 'undefined') return;
        // Aggregate sector composition from member provinces
        const members = data.member_rows || [];
        let agri = 0, indus = 0, serv = 0, taxProd = 0;
        let count = 0;
        members.forEach((m) => {
            const sec = m.sector_composition_pct || {};
            // Also check time_series last row for sector data
            const ts = m.time_series || [];
            const lastRow = ts.length ? ts[ts.length - 1] : {};
            const a = sec.agriculture ?? lastRow.sector_agriculture_pct ?? 0;
            const i = sec.industry ?? lastRow.sector_industry_pct ?? 0;
            const s = sec.services ?? lastRow.sector_services_pct ?? 0;
            const t = sec.tax_product ?? 0;
            if (a || i || s) {
                agri += Number(a);
                indus += Number(i);
                serv += Number(s);
                taxProd += Number(t);
                count++;
            }
        });
        if (count > 0) {
            agri = Math.round((agri / count) * 10) / 10;
            indus = Math.round((indus / count) * 10) / 10;
            serv = Math.round((serv / count) * 10) / 10;
            taxProd = Math.round((taxProd / count) * 10) / 10;
        }
        // Destroy previous if exists
        if (window._mergerSectorChart) window._mergerSectorChart.destroy();
        window._mergerSectorChart = new Chart(canvas, {
            type: 'doughnut',
            data: {
                labels: ['Nông nghiệp', 'CN-XD', 'Dịch vụ', 'Thuế SP'],
                datasets: [{
                    data: [agri, indus, serv, taxProd],
                    backgroundColor: ['#059669', '#2563eb', '#d97706', '#94a3b8'],
                    borderWidth: 1,
                    borderColor: '#fff',
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '55%',
                plugins: {
                    legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10, family: 'Inter' } } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.label}: ${Number(ctx.raw).toFixed(1)}%`,
                        },
                    },
                },
            },
        });
    }

    function renderTable(data) {
        const tbody = byId('merger-comparison-body');
        if (!tbody) return;
        
        let html = '';
        const members = data.member_rows || [];
        const mergedSeries = data.merged_time_series || [];
        
        members.forEach((member) => {
            const series = member.time_series || [];
            const sortedSeries = [...series].sort((a, b) => Number(a.year) - Number(b.year));
            const filteredSeries = sortedSeries.filter(row => Number(row.year) >= 2019);
            
            filteredSeries.forEach((row) => {
                const year = Number(row.year);
                const grdp = Number(row.grdp_billion_vnd_est || 0);
                
                const prevRow = sortedSeries.find(item => Number(item.year) === year - 1);
                const prevGrdp = prevRow ? Number(prevRow.grdp_billion_vnd_est || 0) : 0;
                
                let yoyGrowthText = '--';
                let positive = true;
                if (prevGrdp > 0) {
                    const yoyGrowth = ((grdp - prevGrdp) / prevGrdp) * 100;
                    yoyGrowthText = window.MergerMapData.formatPercent(yoyGrowth);
                    positive = yoyGrowth >= 0;
                }
                
                const mergedRow = mergedSeries.find(item => Number(item.year) === year);
                const mergedTotal = mergedRow ? Number(mergedRow.grdp_billion_vnd_est || 0) : 0;
                const sharePct = mergedTotal > 0 ? (grdp / mergedTotal) * 100 : 0;
                
                html += `
                    <tr class="border-t border-slate-100 hover:bg-slate-50 transition-colors">
                        <td class="px-3 py-2 font-bold text-primary-container">${esc(member.province_name || member.province_code)}</td>
                        <td class="px-3 py-2 font-semibold text-slate-600">${year}</td>
                        <td class="px-3 py-2">${window.MergerMapData.formatNumber(grdp, 1)} tỷ</td>
                        <td class="px-3 py-2 font-bold ${yoyGrowthText === '--' ? 'text-slate-400' : (positive ? 'text-emerald-600' : 'text-red-600')}">${yoyGrowthText}</td>
                        <td class="px-3 py-2">${window.MergerMapData.formatPercent(sharePct)}</td>
                        <td class="px-3 py-2"><span class="rounded-full px-2 py-1 text-[10px] font-black ${yoyGrowthText === '--' ? 'bg-slate-100 text-slate-600' : (positive ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700')}">${yoyGrowthText === '--' ? 'Không có mốc so' : (positive ? 'Tăng trưởng' : 'Suy giảm')}</span></td>
                    </tr>
                `;
            });
        });
        
        tbody.innerHTML = html || '<tr><td colspan="6" class="px-3 py-2 text-center text-slate-400">Không có dữ liệu so sánh.</td></tr>';
    }

    function renderEventCards(data) {
        const container = byId('merger-event-cards');
        if (!container) return;
        const events = Array.isArray(data.events) ? data.events.slice(0, 8) : [];
        if (!events.length) {
            container.innerHTML = '<div class="text-xs text-slate-400">Chưa có sự kiện liên quan.</div>';
            return;
        }
        container.innerHTML = events.map((event) => `
            <div class="rounded-lg border border-slate-200 bg-white p-3">
                <div class="text-sm font-black text-primary-container">${esc(event.event_name_vi || event.event_name || event.event_key)}</div>
                <div class="mt-1 text-xs text-slate-500 line-clamp-2">${esc(event.description_vi || event.description || '')}</div>
                <div class="mt-2 text-[10px] font-bold uppercase text-slate-400">${esc(event.event_type || 'macro')} - ${esc(event.start_date || '')}</div>
            </div>
        `).join('');
    }

    window.addEventListener('macro:province-selected', (event) => {
        const detail = event.detail || {};
        showForProvince(detail.provinceCode, detail.provinceName);
    });

    window.addEventListener('macro:boundary-change', () => {
        if (latestSelection?.provinceCode) {
            showForProvince(latestSelection.provinceCode, latestSelection.provinceName);
        }
    });

    return { showForProvince };
})();

window.MergerAnalysis = MergerAnalysis;
