document.addEventListener("DOMContentLoaded", async () => {
    const tbody = document.getElementById('delinq-tbody');
    if (!tbody) return;

    try {
        const response = await fetch(`${API_BASE}/delinquency`);
        const data = await response.json();

        let rowsHTML = '';
        const items = Array.isArray(data) ? data : data.predictions || [];
        
        items.forEach(c => {
            const risk_probability = c.probability || c.risk_probability || 0;
            const riskBg = risk_probability > 0.8 ? 'bg-error text-white' : 
                           risk_probability > 0.5 ? 'bg-orange-500 text-white' : 
                           'bg-yellow-500 text-white';
            const riskText = risk_probability > 0.8 ? 'bg-error-container text-on-error-container' : 
                           risk_probability > 0.5 ? 'bg-orange-100 text-orange-700' : 
                           'bg-yellow-100 text-yellow-700';

            const widthStr = (risk_probability * 100).toFixed(0) + '%';
            
            rowsHTML += `
            <tr class="hover:bg-slate-50 transition-colors">
                <td class="px-8 py-5 text-slate-500 font-mono">${c.tax_code}</td>
                <td class="px-4 py-5 font-bold text-primary-container">${c.company_name || c.name}</td>
                <td class="px-4 py-5 text-on-surface-variant">${c.cluster || c.industry}</td>
                <td class="px-4 py-5">
                    <div class="flex items-center gap-2">
                        <div class="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                            <div class="h-full ${riskBg}" style="width: ${widthStr}"></div>
                        </div>
                        <span class="px-2 py-0.5 ${riskText} rounded text-[11px] font-bold">${widthStr}</span>
                    </div>
                </td>
                <td class="px-4 py-5 font-bold text-primary-container">${c.predicted_debt_amount ? formatCurrencyCode(c.predicted_debt_amount) : '0'} VNĐ</td>
                <td class="px-8 py-5 text-right">
                    <button class="bg-primary-container text-white px-3 py-1.5 rounded-lg text-xs font-bold hover:opacity-90 transition-all">Ra thông báo</button>
                </td>
            </tr>`;
        });

        tbody.innerHTML = rowsHTML;

    } catch (error) {
        console.error("Delinquency fetch error:", error);
        tbody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-error font-bold">Lỗi tải dữ liệu.</td></tr>';
    }
});
