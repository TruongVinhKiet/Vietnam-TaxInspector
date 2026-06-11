(function () {
    const UI = window.TaxpayerUI;

    function invoiceRow(item) {
        const directionLabel = item.direction === "in" ? "Đầu vào" : "Đầu ra";
        const directionClass = item.direction === "in" ? "text-blue-600" : "text-emerald-600";
        return `
            <tr>
                <td class="px-6 py-3 font-bold ${directionClass}">${directionLabel}</td>
                <td class="px-6 py-3 font-mono font-bold">${UI.escapeHtml(item.invoice_number)}</td>
                <td class="px-6 py-3">${UI.escapeHtml(item.issue_date || "")}</td>
                <td class="px-6 py-3">${UI.escapeHtml(item.partner_name || "Chưa có tên")}</td>
                <td class="px-6 py-3 font-mono">${UI.escapeHtml(item.buyer_tax_code || item.seller_tax_code || "")}</td>
                <td class="px-6 py-3 text-right font-bold">${UI.fmtVnd(item.total_amount || item.amount)}</td>
                <td class="px-6 py-3 text-center">${UI.statusBadge(item.status)}</td>
            </tr>
        `;
    }

    async function loadInvoices() {
        const [log, req] = await Promise.all([
            UI.get("/invoices/log"),
            UI.get("/invoices/einvoice-requirement"),
        ]);
        const body = document.getElementById("invoice-table-body");
        if (body && log.invoices?.length) {
            body.innerHTML = log.invoices.map(invoiceRow).join("");
        }
        UI.panel("invoice-requirement-panel", "Nguong bat buoc hoa don dien tu", "rule", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Trang thai</p>
                    <p class="text-sm font-black text-slate-800">${UI.escapeHtml(req.requirement.label)}</p>
                </div>
                <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Nguong tu nguyen</p>
                    <p class="text-sm font-black text-slate-800">${UI.fmtVnd(req.requirement.thresholds.voluntary_from)}</p>
                </div>
                <div class="p-3 rounded-lg border border-slate-200 bg-slate-50">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Nguong bat buoc</p>
                    <p class="text-sm font-black text-slate-800">${UI.fmtVnd(req.requirement.thresholds.mandatory_from)}</p>
                </div>
            </div>
            <p class="mt-3 text-[11px] text-slate-500">${UI.escapeHtml(req.requirement.action)}</p>
        `);
    }

    window.issueInvoice = async function issueInvoice() {
        try {
            const payload = {
                buyer_tax_code: UI.readValue("buyer-mst"),
                buyer_name: UI.readValue("buyer-name"),
                item_description: UI.readValue("item-desc"),
                vat_rate: Number(UI.readValue("item-vat", 8)),
                unit_price: Number(UI.readValue("item-price", 0)),
                quantity: Number(UI.readValue("item-qty", 1)),
            };
            const data = await UI.post("/invoices/issue", payload);
            UI.toast(`Đã phát hành ${data.invoice.invoice_number}`);
            await loadInvoices();
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.scanInvoice = async function scanInvoice() {
        try {
            const seller = UI.readValue("scan-mst");
            const data = await UI.post("/invoices/scan", { seller_tax_code: seller, tax_code: seller });
            const flags = data.scan.risk_flags?.length ? data.scan.risk_flags.join(", ") : "Không phát hiện cờ rủi ro sandbox";
            UI.panel("invoice-scan-result-panel", "Kết quả rà soát hóa đơn đầu vào", "fact_check", `
                <div class="flex items-center justify-between gap-3">
                    <div>
                        <p class="font-bold text-slate-800">${UI.escapeHtml(data.scan.message)}</p>
                        <p class="text-[11px] text-slate-500 mt-1">${UI.escapeHtml(flags)}</p>
                    </div>
                    ${UI.statusBadge(data.scan.status)}
                </div>
            `);
            await loadInvoices();
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.runSupplierTrustAssessment = async function runSupplierTrustAssessment() {
        try {
            const data = await UI.get("/intelligence/pagerank-supplier-trust");
            const resultContainer = document.getElementById("supplier-trust-result");
            if (!resultContainer) return;
            
            resultContainer.classList.remove("hidden");
            
            const tbody = document.getElementById("supplier-trust-body");
            tbody.innerHTML = (data.suppliers || []).map(sup => `
                <tr>
                    <td class="px-4 py-2 font-medium">
                        <div>${UI.escapeHtml(sup.name)}</div>
                        <div class="text-[9px] text-slate-400 font-mono">MST: ${UI.escapeHtml(sup.tax_code || '---')}</div>
                    </td>
                    <td class="px-4 py-2 text-right font-mono font-bold">${UI.fmtVnd(sup.total_amount)}</td>
                    <td class="px-4 py-2 text-center font-mono">${sup.invoice_count} HĐ (${sup.bank_confirmed} bank)</td>
                    <td class="px-4 py-2 text-center font-mono font-bold text-indigo-600">${sup.trust_score}/100</td>
                    <td class="px-4 py-2 text-center">
                        <span class="px-2 py-0.5 rounded text-[9px] font-bold uppercase ${
                            sup.trust_tier === 'A' ? 'bg-emerald-100 text-emerald-800' :
                            sup.trust_tier === 'B' ? 'bg-blue-100 text-blue-800' :
                            sup.trust_tier === 'C' ? 'bg-amber-100 text-amber-800' : 'bg-rose-100 text-rose-800'
                        }">
                            Hạng ${sup.trust_tier}
                        </span>
                    </td>
                </tr>
            `).join("");
            
            document.getElementById("supplier-trust-alert").textContent = data.explanation?.counterfactual?.verify_suppliers || "Hệ sinh thái nhà cung cấp an toàn.";
            UI.toast("Đã hoàn tất đánh giá tín nhiệm nhà cung cấp bằng PageRank.", "success");
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.runSpectralEvasionCascade = async function runSpectralEvasionCascade() {
        try {
            const data = await UI.get("/intelligence/spectral-cascade");
            const resultContainer = document.getElementById("gnn-spectral-result");
            if (!resultContainer) return;

            resultContainer.classList.remove("hidden");

            // Gap & loops
            document.getElementById("res-gnn-gap").textContent = Number(data.spectral_gap).toFixed(6);
            
            const loopEl = document.getElementById("res-gnn-loops");
            if (data.circular_invoicing_loops > 0) {
                loopEl.textContent = "PHÁT HIỆN VÒNG LẶP!";
                loopEl.className = "text-lg font-black text-rose-600 uppercase";
            } else {
                loopEl.textContent = "An toàn";
                loopEl.className = "text-lg font-black text-emerald-600 uppercase";
            }

            // Cascade propagation body
            const cascadeBody = document.getElementById("gnn-cascade-body");
            cascadeBody.innerHTML = (data.risk_cascade_propagation || []).map(node => {
                const colorClass = node.evasion_risk_exposure > 50 ? 'bg-rose-500' : node.evasion_risk_exposure > 25 ? 'bg-amber-500' : 'bg-emerald-500';
                return `
                    <tr>
                        <td class="px-4 py-2 font-mono font-medium">${UI.escapeHtml(node.tax_code)}</td>
                        <td class="px-4 py-2">
                            <div class="flex items-center gap-2">
                                <span class="font-bold font-mono text-[10px] w-8">${node.evasion_risk_exposure}%</span>
                                <div class="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                                    <div class="${colorClass} h-1.5 rounded-full" style="width: ${node.evasion_risk_exposure}%"></div>
                                </div>
                            </div>
                        </td>
                        <td class="px-4 py-2 text-center font-mono font-bold">${node.connections} cạnh</td>
                    </tr>
                `;
            }).join("");

            // Collusion similarity mapping
            const collusionContainer = document.getElementById("gnn-collusion-container");
            const aaEntries = Object.entries(data.adamic_adar_collusion || {});
            if (aaEntries.length === 0) {
                collusionContainer.innerHTML = `<p class="text-slate-400 font-medium italic">Không phát hiện liên kết trùng khớp địa chỉ hoặc tần suất bất thường.</p>`;
            } else {
                collusionContainer.innerHTML = `
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-2 mt-1">
                        ${aaEntries.map(([pair, score]) => `
                            <div class="bg-white p-2 rounded border border-slate-100 flex justify-between items-center">
                                <span class="font-mono text-slate-600 font-semibold">${pair}</span>
                                <span class="px-2 py-0.5 bg-amber-50 text-amber-700 font-bold font-mono rounded border border-amber-200">Score: ${score}</span>
                            </div>
                        `).join("")}
                    </div>
                `;
            }

            document.getElementById("gnn-verdict").textContent = `${data.verdict} Đề xuất: ${data.explanation?.counterfactual?.verify_path || '---'}`;
            UI.toast("Phân tích đồ thị trốn thuế vòng lặp GNN hoàn tất.", "success");
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    document.addEventListener("DOMContentLoaded", () => UI.boot(loadInvoices));
})();

