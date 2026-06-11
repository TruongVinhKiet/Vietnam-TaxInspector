(function () {
    const UI = window.TaxpayerUI;

    function addMessage(role, html) {
        const box = document.getElementById("chat-messages");
        if (!box) return;
        const isUser = role === "user";
        const wrap = document.createElement("div");
        wrap.className = `flex gap-3 ${isUser ? "justify-end ml-auto" : ""} max-w-[88%]`;
        wrap.innerHTML = isUser
            ? `<div class="bg-emerald-500 text-white p-3 rounded-2xl rounded-tr-none">${html}</div>`
            : `<div class="w-7 h-7 bg-slate-100 rounded-full flex items-center justify-center font-bold text-slate-500 flex-shrink-0">AI</div><div class="bg-slate-100 p-3 rounded-2xl rounded-tl-none text-slate-700">${html}</div>`;
        box.appendChild(wrap);
        box.scrollTop = box.scrollHeight;
    }

    window.sendChatMessage = async function sendChatMessage() {
        const input = document.getElementById("chat-input");
        const message = input?.value?.trim();
        if (!message) return;
        input.value = "";
        addMessage("user", UI.escapeHtml(message));
        try {
            const data = await UI.post("/legal/chat", { message });
            const citations = (data.citations || []).map((item) => `<a class="text-emerald-700 font-bold underline" target="_blank" href="${UI.escapeHtml(item.source_url)}">${UI.escapeHtml(item.key)}</a>`).join(" · ");
            addMessage("ai", `${UI.escapeHtml(data.answer)}${citations ? `<div class="mt-2 text-[10px] text-slate-500">Nguồn: ${citations}</div>` : ""}`);
        } catch (e) {
            addMessage("ai", `<span class="text-rose-600">${UI.escapeHtml(e.message)}</span>`);
        }
    };

    window.handleKeyPress = function handleKeyPress(event) {
        if (event.key === "Enter") {
            event.preventDefault();
            window.sendChatMessage();
        }
    };

    async function loadLegalPanels() {
        const [rates, docs, comparison] = await Promise.all([
            UI.get("/legal/rates"),
            UI.get("/legal/documents"),
            UI.get("/legal/hkd-vs-llc"),
        ]);
        UI.panel("legal-rate-lookup-panel", "Tra cứu tỷ lệ thuế theo ngành nghề", "percent", `
            <div class="flex gap-2">
                <input id="legal-rate-query" class="flex-1 rounded-lg border-slate-200 text-xs" placeholder="Nhập ngành nghề, ISIC hoặc mặt hàng...">
                <button id="legal-rate-btn" class="px-3 py-2 rounded-lg bg-[#002147] text-white text-[10px] font-bold">Tra cứu</button>
            </div>
            <div id="legal-rate-results" class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                ${rates.rates.slice(0, 4).map(rateCard).join("")}
            </div>
        `);
        document.getElementById("legal-rate-btn").onclick = async () => {
            const q = UI.readValue("legal-rate-query");
            const data = await UI.get(`/legal/rates?query=${encodeURIComponent(q)}`);
            document.getElementById("legal-rate-results").innerHTML = data.rates.map(rateCard).join("") || `<p class="text-slate-400">Không tìm thấy.</p>`;
        };
        UI.panel("legal-updates-panel", "Văn bản mới đang theo dõi", "newspaper", `
            <div class="space-y-2">
                ${(docs.documents || []).slice(0, 6).map((doc) => `
                    <a target="_blank" href="${UI.escapeHtml(doc.source_url)}" class="block p-3 rounded-lg bg-slate-50 border border-slate-200 hover:bg-slate-100">
                        <p class="font-bold text-slate-800">${UI.escapeHtml(doc.title)}</p>
                        <p class="text-[10px] text-slate-500 mt-1">${UI.escapeHtml(doc.category)} · ${UI.escapeHtml(doc.article_ref || "")}</p>
                    </a>
                `).join("")}
            </div>
        `);
        UI.panel("legal-hkd-llc-panel", "So sánh HKD và Công ty TNHH", "compare_arrows", `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Thuế HKD ước tính</p>
                    <p class="font-black text-slate-800">${UI.fmtVnd(comparison.comparison.hkd_estimated_tax)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Thuế TNHH ước tính</p>
                    <p class="font-black text-slate-800">${UI.fmtVnd(comparison.comparison.llc_total_tax)}</p>
                </div>
                <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                    <p class="text-[9px] uppercase font-bold text-slate-400">Lợi nhuận</p>
                    <p class="font-black text-slate-800">${UI.fmtVnd(comparison.comparison.profit)}</p>
                </div>
            </div>
        `);
    }

    function rateCard(rate) {
        return `
            <div class="p-3 rounded-lg bg-slate-50 border border-slate-200">
                <p class="font-bold text-slate-800">${UI.escapeHtml(rate.name)}</p>
                <p class="text-[10px] text-slate-500 mt-1">GTGT ${rate.gtgt_rate_pct}% · TNCN ${rate.tncn_rate_pct}% · ISIC ${UI.escapeHtml(rate.isic_hint)}</p>
            </div>
        `;
    }

    window.startForensicDebate = async function startForensicDebate() {
        const topic = document.getElementById("debate-topic").value;
        const revenue = Number(document.getElementById("debate-revenue").value) || 0;
        const expenses = Number(document.getElementById("debate-expenses").value) || 0;

        try {
            const data = await UI.post("/intelligence/debate", { topic, revenue, expenses });
            
            document.getElementById("debate-hkd-points").innerHTML = (data.hkd_points || []).map(p => `<li>${UI.escapeHtml(p)}</li>`).join("");
            document.getElementById("debate-llc-points").innerHTML = (data.llc_points || []).map(p => `<li>${UI.escapeHtml(p)}</li>`).join("");

            const dialogueEl = document.getElementById("debate-dialogue");
            dialogueEl.innerHTML = (data.rounds || []).map((r, i) => {
                const isLeft = i % 2 === 0;
                return `
                    <div class="flex gap-2.5 items-start ${isLeft ? '' : 'flex-row-reverse text-right'}">
                        <div class="w-8 h-8 rounded-full ${isLeft ? 'bg-rose-100 text-rose-700' : 'bg-emerald-100 text-emerald-700'} flex items-center justify-center font-bold flex-shrink-0">
                            <span class="material-symbols-outlined text-sm">${UI.escapeHtml(r.avatar || 'smart_toy')}</span>
                        </div>
                        <div class="max-w-[75%] space-y-1">
                            <div class="text-[9px] font-black text-slate-400">${UI.escapeHtml(r.speaker)} (${UI.escapeHtml(r.role)})</div>
                            <div class="p-2.5 rounded-lg ${isLeft ? 'bg-rose-50/50 text-rose-950 rounded-tl-none border border-rose-100' : 'bg-emerald-50/50 text-emerald-950 rounded-tr-none border border-emerald-100'} text-[11px] leading-relaxed">
                                ${UI.escapeHtml(r.statement)}
                            </div>
                        </div>
                    </div>
                `;
            }).join("");

            document.getElementById("debate-winner").textContent = data.winner;
            document.getElementById("debate-verdict").textContent = data.verdict;
            document.getElementById("debate-gauge-bar").style.width = `${data.gauge_pct}%`;

            document.getElementById("debate-board").classList.remove("hidden");
            dialogueEl.scrollTop = dialogueEl.scrollHeight;
        } catch (e) {
            UI.toast(e.message, "error");
        }
    };

    document.addEventListener("DOMContentLoaded", () => UI.boot(loadLegalPanels));
})();
