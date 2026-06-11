(function () {
    const UI = window.TaxpayerUI;

    async function loadGrowth() {
        const data = await UI.get("/growth/readiness");
        UI.panel("growth-readiness-panel", "Sẵn sàng thay đổi quy mô", "trending_up", `
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Nhóm HKD</p>
                    <p class="font-black text-slate-800">${UI.escapeHtml(data.household_group.label)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Hóa đơn điện tử</p>
                    <p class="font-black text-slate-800">${UI.escapeHtml(data.einvoice.status)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Tồn kho đầu kỳ</p>
                    <p class="font-black text-slate-800">${data.inventory_opening_required ? "Bắt buộc" : "Chưa bắt buộc"}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">TNHH ước tính</p>
                    <p class="font-black text-slate-800">${UI.fmtVnd(data.llc_comparison.llc_total_tax)}</p>
                </div>
            </div>
        `);
        UI.panel("growth-extra-events-panel", "Thay đổi địa điểm, ngành nghề và tồn kho", "edit_location_alt", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <button id="relocate-btn" class="py-2 rounded-lg bg-slate-100 text-slate-700 text-[10px] font-bold">Thông báo đổi địa điểm</button>
                <button id="industry-btn" class="py-2 rounded-lg bg-slate-100 text-slate-700 text-[10px] font-bold">Bổ sung ngành nghề</button>
                <button id="inventory-btn" class="py-2 rounded-lg bg-[#002147] text-white text-[10px] font-bold">Khai tồn kho 31/12</button>
            </div>
            <div id="growth-event-result" class="mt-3 text-[11px] text-slate-500"></div>
        `);
        document.getElementById("relocate-btn").onclick = () => submitEvent("relocate", { address: "Địa điểm mới sandbox" });
        document.getElementById("industry-btn").onclick = () => submitEvent("industry_change", { industry: "service" });
        document.getElementById("inventory-btn").onclick = () => submitEvent("opening_inventory", { inventory_date: "2025-12-31", note: "Bảng kê tồn kho đầu kỳ" });
        
        await loadEcommerceReconciliation();
    }

    async function loadEcommerceReconciliation() {
        try {
            const data = await UI.get("/intelligence/ecommerce/reconcile");
            if (!data || data.status !== "success") return;

            const platHtml = Object.entries(data.platform_summary).map(([plat, s]) => `
                <div class="p-4 rounded-xl bg-slate-50 border border-slate-100 space-y-2.5">
                    <div class="flex justify-between items-center border-b border-slate-200/60 pb-2">
                        <span class="font-bold text-slate-800 text-xs">${UI.escapeHtml(plat)}</span>
                        <span class="text-[9px] px-2 py-0.5 font-bold rounded bg-indigo-50 text-indigo-700">
                            Khớp ${s.matched_count} đơn
                        </span>
                    </div>
                    <div class="grid grid-cols-2 gap-2 text-[10px]">
                        <div>
                            <p class="text-slate-400 font-medium">Doanh thu sàn</p>
                            <p class="font-bold text-slate-700 text-xs">${UI.fmtVnd(s.gross)}</p>
                        </div>
                        <div>
                            <p class="text-slate-400 font-medium">Phí hoa hồng/ship</p>
                            <p class="font-bold text-slate-700 text-xs">${UI.fmtVnd(s.commission + s.shipping)}</p>
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-2 text-[10px] pt-1.5 border-t border-slate-100">
                        <div>
                            <p class="text-slate-400 font-medium">Thực nhận (Net)</p>
                            <p class="font-bold text-emerald-600 text-xs">${UI.fmtVnd(s.net)}</p>
                        </div>
                        <div>
                            <p class="text-slate-400 font-medium">Chưa khớp (Rủi ro)</p>
                            <p class="font-bold ${s.unmatched_net > 0 ? 'text-rose-600' : 'text-slate-500'} text-xs">${UI.fmtVnd(s.unmatched_net)}</p>
                        </div>
                    </div>
                </div>
            `).join("");

            let anomalyHtml = "";
            if (data.anomalies && data.anomalies.length > 0) {
                anomalyHtml = `
                    <div class="space-y-3">
                        <p class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Cảnh báo chênh lệch & rủi ro phát hiện:</p>
                        ${data.anomalies.map(a => `
                            <div class="p-3.5 rounded-xl border ${a.severity === 'high' ? 'bg-rose-50/70 border-rose-200 text-rose-950' : 'bg-amber-50/70 border-amber-200 text-amber-950'} space-y-1.5 text-xs">
                                <div class="flex items-center gap-2 font-bold">
                                    <span class="material-symbols-outlined text-sm ${a.severity === 'high' ? 'text-rose-600' : 'text-amber-600'}">
                                        ${a.severity === 'high' ? 'report' : 'warning'}
                                    </span>
                                    <span>${UI.escapeHtml(a.title)}</span>
                                    <span class="text-[9px] px-1.5 py-0.5 rounded uppercase font-extrabold ${a.severity === 'high' ? 'bg-rose-100 text-rose-800' : 'bg-amber-100 text-amber-800'} ml-auto">
                                        ${a.severity === 'high' ? 'Nghiêm trọng' : 'Trung bình'}
                                    </span>
                                </div>
                                <p class="text-[11px] leading-relaxed text-slate-600">${UI.escapeHtml(a.description)}</p>
                                <div class="pt-1.5 border-t ${a.severity === 'high' ? 'border-rose-200/50' : 'border-amber-200/50'}">
                                    <p class="text-[9px] font-bold uppercase text-slate-500 mb-1">Hành động khắc phục đề xuất:</p>
                                    <ul class="list-disc pl-4 text-[10px] space-y-1 text-slate-600">
                                        ${a.suggested_actions.map(act => `<li>${UI.escapeHtml(act)}</li>`).join("")}
                                    </ul>
                                </div>
                            </div>
                        `).join("")}
                    </div>
                `;
            } else {
                anomalyHtml = `
                    <div class="p-4 rounded-xl bg-emerald-50 border border-emerald-100 text-emerald-800 flex items-center gap-2.5 text-xs font-semibold">
                        <span class="material-symbols-outlined text-lg">check_circle</span>
                        <span>Hoàn hảo: Tất cả các dòng tiền đối soát từ Shopee, Lazada và TikTok Shop khớp 100% với sao kê ngân hàng thụ hưởng.</span>
                    </div>
                `;
            }

            UI.panel("growth-ecommerce-reconciliation", "Đối soát Doanh thu & Phí sàn E-Commerce", "shopping_bag", `
                <div class="space-y-6">
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-3 bg-slate-50 p-4 rounded-xl border border-slate-200/80">
                        <div>
                            <p class="text-[9px] uppercase font-bold text-slate-400">Tổng doanh thu sàn (Gross)</p>
                            <p class="font-black text-slate-800 text-sm mt-0.5">${UI.fmtVnd(data.total_gross)}</p>
                        </div>
                        <div>
                            <p class="text-[9px] uppercase font-bold text-slate-400">Khấu trừ phí sàn & ship</p>
                            <p class="font-black text-slate-800 text-sm mt-0.5">${UI.fmtVnd(data.total_fees)}</p>
                        </div>
                        <div>
                            <p class="text-[9px] uppercase font-bold text-slate-400">Doanh thu nhận ròng (Net)</p>
                            <p class="font-black text-emerald-600 text-sm mt-0.5">${UI.fmtVnd(data.total_net)}</p>
                        </div>
                        <div>
                            <p class="text-[9px] uppercase font-bold text-slate-400">Tỷ lệ khớp ngân hàng</p>
                            <p class="font-black text-indigo-600 text-sm mt-0.5">${data.matched_ratio}%</p>
                        </div>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        ${platHtml}
                    </div>

                    ${anomalyHtml}
                </div>
            `, { prepend: true });
        } catch (e) {
            console.error("[ECOMMERCE RECON ERROR]", e);
        }
    }

    async function submitEvent(event_type, payload = {}) {
        const data = await UI.post("/growth/event", { event_type, ...payload });
        const result = document.getElementById("growth-event-result");
        if (result) result.textContent = `Đã ghi nhận: ${data.event.external_ref}`;
        UI.toast("Đã gửi yêu cầu sandbox.");
    }

    window.requestUpgrade = async function requestUpgrade() {
        try {
            await submitEvent("upgrade_to_llc", { target_model: "limited_liability_company" });
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.requestClosure = async function requestClosure() {
        try {
            const type = UI.readValue("stop-type", "suspend");
            const start = UI.readValue("stop-start") || new Date().toISOString().slice(0, 10);
            const end = UI.readValue("stop-end") || null;
            await submitEvent(type === "close" ? "closure" : "temporary_suspension", {
                start_date: start,
                end_date: end,
            });
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    window.runMarkovChainPrediction = async function runMarkovChainPrediction() {
        try {
            const data = await UI.get("/intelligence/markov-chain-prediction");
            const resultContainer = document.getElementById("markov-result-container");
            if (!resultContainer) return;
            
            resultContainer.classList.remove("hidden");
            
            document.getElementById("markov-current-state").textContent = data.current_state;
            
            const steadyProbs = document.getElementById("markov-steady-probs");
            steadyProbs.innerHTML = Object.entries(data.steady_state || {}).map(([state, prob]) => `
                <div class="px-3 py-1 bg-white border border-slate-200 rounded-lg">
                    <span class="font-semibold text-slate-500 uppercase tracking-wider text-[9px]">${state}:</span>
                    <span class="font-mono font-bold text-slate-800">${(prob * 100).toFixed(1)}%</span>
                </div>
            `).join("");
            
            const forecastBody = document.getElementById("markov-forecast-body");
            const step_probs = data.step_probabilities || [];
            const trajectory = data.trajectory || [];
            forecastBody.innerHTML = step_probs.map((step, idx) => {
                const growthVal = step.growth;
                const stableVal = step.stable;
                const declineVal = step.decline;
                // trajectory index 0 is current state, index step is forecast for that step
                const mostLikely = trajectory[step.step] || "stable";
                return `
                    <tr>
                        <td class="px-4 py-2 font-mono font-bold">Tháng ${step.step}</td>
                        <td class="px-4 py-2 text-center font-mono">${(growthVal * 100).toFixed(1)}%</td>
                        <td class="px-4 py-2 text-center font-mono">${(stableVal * 100).toFixed(1)}%</td>
                        <td class="px-4 py-2 text-center font-mono">${(declineVal * 100).toFixed(1)}%</td>
                        <td class="px-4 py-2">
                            <span class="px-2 py-0.5 rounded text-[9px] font-bold uppercase ${
                                mostLikely === 'growth' ? 'bg-emerald-100 text-emerald-800' :
                                mostLikely === 'stable' ? 'bg-slate-100 text-slate-700' : 'bg-rose-100 text-rose-800'
                            }">
                                ${mostLikely}
                            </span>
                        </td>
                    </tr>
                `;
            }).join("");
            
            const recommendVal = data.explanation?.counterfactual?.improve_margin || "Hội tụ về trạng thái dài hạn an toàn.";
            document.getElementById("markov-recommendation").textContent = recommendVal;
            UI.toast("Đã hoàn tất dự báo chuyển trạng thái bằng Chuỗi Markov.", "success");
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    document.addEventListener("DOMContentLoaded", () => UI.boot(loadGrowth));
})();

