function escapeHtml(value) {
    const str = value === null || value === undefined ? '' : String(value);
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}


function toSafeNumber(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}


function renderDelinquencyRows(tbody, items) {
    if (!Array.isArray(items) || items.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center py-6 text-slate-400 font-bold">Không có dữ liệu dự báo.</td></tr>';
        return;
    }

    const rowsHTML = items.map((item) => {
        const riskProbabilityRaw = toSafeNumber(item.probability ?? item.risk_probability, 0);
        const riskProbability = Math.max(0, Math.min(1, riskProbabilityRaw));
        const riskBg = riskProbability > 0.8
            ? 'bg-error text-white'
            : riskProbability > 0.5
                ? 'bg-orange-500 text-white'
                : 'bg-yellow-500 text-white';
        const riskText = riskProbability > 0.8
            ? 'bg-error-container text-on-error-container'
            : riskProbability > 0.5
                ? 'bg-orange-100 text-orange-700'
                : 'bg-yellow-100 text-yellow-700';

        const widthPct = (riskProbability * 100).toFixed(0);
        const widthStr = `${widthPct}%`;

        const taxCode = escapeHtml(item.tax_code || '---');
        const companyName = escapeHtml(item.company_name || item.name || '---');
        const cluster = escapeHtml(item.cluster || item.industry || '---');
        const debtAmount = toSafeNumber(item.predicted_debt_amount, 0);

        return `
            <tr class="hover:bg-slate-50 transition-colors">
                <td class="px-8 py-5 text-slate-500 font-mono">${taxCode}</td>
                <td class="px-4 py-5 font-bold text-primary-container">${companyName}</td>
                <td class="px-4 py-5 text-on-surface-variant">${cluster}</td>
                <td class="px-4 py-5">
                    <div class="flex items-center gap-2">
                        <div class="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                            <div class="h-full ${riskBg}" style="width: ${widthStr}"></div>
                        </div>
                        <span class="px-2 py-0.5 ${riskText} rounded text-[11px] font-bold">${widthStr}</span>
                    </div>
                </td>
                <td class="px-4 py-5 font-bold text-primary-container">${formatCurrencyCode(debtAmount)} VNĐ</td>
                <td class="px-8 py-5 text-right">
                    <button class="bg-primary-container text-white px-3 py-1.5 rounded-lg text-xs font-bold hover:opacity-90 transition-all">Ra thông báo</button>
                </td>
            </tr>`;
    }).join('');

    tbody.innerHTML = rowsHTML;
}


document.addEventListener('DOMContentLoaded', async () => {
    const tbody = document.getElementById('delinq-tbody');
    if (!tbody) return;

    try {
        const response = await secureFetch(`${API_BASE}/delinquency`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        const items = Array.isArray(data) ? data : (data.predictions || []);
        renderDelinquencyRows(tbody, items);
    } catch (error) {
        console.error('Delinquency fetch error:', error);
        tbody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-error font-bold">Lỗi tải dữ liệu.</td></tr>';
    }
});
