(function () {
    const UI = window.TaxpayerUI;
    let lastFilingId = null;
    let lastPaymentRef = null;

    function currentDraftPayload(extra = {}) {
        const revenue = Number(UI.readValue("file-gtgt-rev", 0));
        return {
            revenue,
            gtgt_revenue: revenue,
            tncn_revenue: Number(UI.readValue("file-tncn-rev", revenue)),
            period: "2026-Q1",
            form_code: "01/CNKD",
            industry: "commerce",
            ...extra,
        };
    }

    async function ensureDraft(extra = {}) {
        const data = await UI.post("/filings/draft", currentDraftPayload(extra));
        lastFilingId = data.filing.id;
        updateTaxNumbers(data.filing);
        return data.filing;
    }

    function updateTaxNumbers(filing) {
        const gtgt = document.getElementById("file-gtgt-val");
        const tncn = document.getElementById("file-tncn-val");
        const total = document.getElementById("file-total-val");
        if (gtgt) gtgt.textContent = UI.fmtVnd(filing.gtgt_tax);
        if (tncn) tncn.textContent = UI.fmtVnd(filing.tncn_tax);
        if (total) total.textContent = UI.fmtVnd(filing.total_tax);
        const payAmount = document.getElementById("pay-amount");
        if (payAmount) payAmount.value = Math.round(Number(filing.total_tax || 0));
    }

    async function loadFilings() {
        const data = await UI.get("/filings/status");
        const rows = data.filings || [];
        UI.panel("filing-status-panel", "Trạng thái hồ sơ và tờ khai bổ sung", "folder_managed", `
            <div class="overflow-x-auto">
                <table class="w-full text-left text-xs">
                    <thead class="text-[9px] uppercase text-slate-400">
                        <tr><th class="py-2">Mẫu</th><th>Kỳ</th><th>Loại</th><th>Số thuế</th><th>Trạng thái</th></tr>
                    </thead>
                    <tbody class="divide-y divide-slate-100">
                        ${rows.slice(0, 6).map((item) => `
                            <tr>
                                <td class="py-2 font-mono font-bold">${UI.escapeHtml(item.form_code)}</td>
                                <td>${UI.escapeHtml(item.period)}</td>
                                <td>${UI.escapeHtml(item.filing_type)}</td>
                                <td>${UI.fmtVnd(item.total_tax)}</td>
                                <td>${UI.statusBadge(item.status)}</td>
                            </tr>
                        `).join("") || `<tr><td colspan="5" class="py-3 text-slate-400">Chưa có tờ khai nào.</td></tr>`}
                    </tbody>
                </table>
            </div>
            <button id="filing-amend-btn" class="mt-3 px-3 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-[10px] font-bold">
                Tạo tờ khai bổ sung từ bản mới nhất
            </button>
        `);
        const btn = document.getElementById("filing-amend-btn");
        if (btn) btn.onclick = amendLatest;
    }

    async function amendLatest() {
        try {
            const filing = await ensureDraft({ filing_type: "amendment", idempotency_key: `amend-${Date.now()}` });
            UI.toast(`Da tao to khai bo sung #${filing.id}`);
            await loadFilings();
        } catch (e) {
            UI.toast(e.message, "error");
        }
    }

    window.exportXml = async function exportXml() {
        try {
            const filing = await ensureDraft();
            window.open(`${API_BASE}/taxpayer/filings/${filing.id}/xml`, "_blank");
            UI.toast("Da tao file XML to khai.");
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.signFiling = async function signFiling() {
        try {
            const filing = await ensureDraft();
            const data = await UI.post(`/filings/${filing.id}/submit`, { signature: "sandbox-signature" });
            UI.toast(data.gateway.message || "Da nop to khai.");
            await loadFilings();
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.generateQr = async function generateQr() {
        try {
            const amount = Number(UI.readValue("pay-amount", 0));
            const data = await UI.post("/filings/payment-qr", { filing_id: lastFilingId, period: "2026-Q1", amount });
            lastPaymentRef = data.qr.payment_ref;
            const qrBox = document.getElementById("qr-box");
            if (qrBox) {
                qrBox.classList.remove("hidden");
                const note = qrBox.querySelector("p");
                if (note) note.textContent = data.qr.qr_payload;
            }
            UI.toast("Da tao QR thanh toan sandbox.");
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.confirmPayment = async function confirmPayment() {
        try {
            const data = await UI.post("/filings/payment-confirm", { payment_ref: lastPaymentRef });
            UI.toast(data.idempotent ? "Thanh toan da duoc xac nhan truoc do." : "Da xac nhan thanh toan.");
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.loadProofs = async function loadProofs() {
        try {
            const data = await UI.get("/accounting/documents");
            const proofs = data.documents || [];
            const listEl = document.getElementById("proofs-list");
            if (!listEl) return;
            
            const proofDocs = proofs.filter(p => ["evidence", "invoice", "bank_proof"].includes(p.doc_type));
            if (proofDocs.length === 0) {
                listEl.innerHTML = `<div class="text-slate-400 italic">Chưa có chứng từ đính kèm.</div>`;
                return;
            }

            listEl.innerHTML = proofDocs.map(p => {
                const meta = p.metadata_json || {};
                const billingId = meta.billing_id || (p.sha256 ? "HD-" + p.sha256.substring(0, 8).toUpperCase() : `HD-${p.id}`);
                const amount = Number(meta.amount || 0);
                return `<div class="flex justify-between items-center py-0.5 border-b border-slate-100">
                    <span class="truncate max-w-[120px]">${UI.escapeHtml(billingId)}</span>
                    <span class="font-bold text-slate-800">${UI.fmtVnd(amount)}</span>
                </div>`;
            }).join("");
        } catch (e) {
            console.error("[LOAD PROOFS ERROR]", e);
        }
    };

    window.simulateUploadProof = async function simulateUploadProof() {
        const billingId = document.getElementById("proof-billing-id").value.trim();
        const amountStr = document.getElementById("proof-amount").value.trim();
        if (!billingId || !amountStr) {
            UI.toast("Vui lòng điền mã hóa đơn và số tiền.", "error");
            return;
        }

        const amount = Number(amountStr);
        if (isNaN(amount) || amount <= 0) {
            UI.toast("Số tiền phải lớn hơn 0.", "error");
            return;
        }

        try {
            const fakeFileContent = `Simulated Invoice metadata:\nBilling ID: ${billingId}\nAmount: ${amount}`;
            const blob = new Blob([fakeFileContent], { type: "text/plain" });
            const formData = new FormData();
            formData.append("file", blob, `${billingId}_invoice.txt`);
            formData.append("doc_type", "evidence");
            formData.append("billing_id", billingId);
            formData.append("amount", amount.toString());

            const response = await secureFetch(`${API_BASE}/taxpayer/accounting/documents`, {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status}`);
            }

            UI.toast("Đã đính kèm chứng từ thành công.");
            document.getElementById("proof-billing-id").value = "";
            document.getElementById("proof-amount").value = "";
            await loadProofs();
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.runPreFilingAIValidation = async function runPreFilingAIValidation() {
        if (!lastFilingId) {
            UI.toast("Vui lòng khởi tạo tờ khai nháp trước.", "error");
            return;
        }

        try {
            const data = await UI.post(`/filings/${lastFilingId}/validate-proof`);
            const box = document.getElementById("ai-validation-box");
            if (!box) return;

            box.classList.remove("hidden");
            
            let statusClass = "bg-emerald-50 border-emerald-200 text-emerald-950";
            let statusText = "HỢP LỆ (KHỚP 100%)";
            if (data.validation_status === "warning") {
                statusClass = "bg-amber-50 border-amber-200 text-amber-950";
                statusText = "CẢNH BÁO (CÓ CHÊNH LỆCH NHẸ)";
            } else if (data.validation_status === "invalid") {
                statusClass = "bg-rose-50 border-rose-200 text-rose-950";
                statusText = "RỦI RO CAO (THIẾU MINH CHỨNG)";
            }

            let issuesHtml = "";
            if (data.issues && data.issues.length > 0) {
                issuesHtml = data.issues.map(i => `
                    <div class="p-2.5 rounded border ${i.severity === 'high' ? 'bg-rose-100/50 border-rose-200 text-rose-900' : 'bg-amber-100/50 border-amber-200 text-amber-900'} space-y-1">
                        <div class="flex items-center gap-1 font-bold text-[10px]">
                            <span class="material-symbols-outlined text-xs">info</span>
                            <span>${UI.escapeHtml(i.title)}</span>
                        </div>
                        <p class="text-[10px] text-slate-700 leading-tight">${UI.escapeHtml(i.message)}</p>
                        <p class="text-[9px] text-emerald-700 font-semibold italic">Đề xuất: ${UI.escapeHtml(i.suggestion)}</p>
                    </div>
                `).join("");
            } else {
                issuesHtml = `<div class="text-[10px] text-emerald-800 font-semibold">✓ Không phát hiện bất kỳ chênh lệch hay mã hóa đơn sai lệch nào. Đủ điều kiện nộp tờ khai.</div>`;
            }

            box.className = `p-3 rounded-lg border text-[11px] space-y-2.5 ${statusClass}`;
            box.innerHTML = `
                <div class="flex justify-between items-center font-bold">
                    <span>KẾT QUẢ AI RÀ SOÁT:</span>
                    <span class="text-[9px] px-2 py-0.5 rounded font-extrabold border bg-white">${statusText}</span>
                </div>
                <div class="grid grid-cols-2 gap-2 text-[10px] border-b pb-2/60 border-current/20">
                    <div>
                        <p class="opacity-75">Doanh thu kê khai:</p>
                        <p class="font-bold">${UI.fmtVnd(data.declared_revenue)}</p>
                    </div>
                    <div>
                        <p class="opacity-75">Minh chứng đính kèm:</p>
                        <p class="font-bold">${UI.fmtVnd(data.total_proof_amount)}</p>
                    </div>
                </div>
                <div class="space-y-2">
                    ${issuesHtml}
                </div>
            `;
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    document.addEventListener("DOMContentLoaded", () => UI.boot(async () => {
        await ensureDraft();
        await loadFilings();
        await loadProofs();
    }));
})();
