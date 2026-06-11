(function () {
    const UI = window.TaxpayerUI;
    const Registry = window.TaxpayerAIRegistry;
    if (!UI || !Registry) return;

    function renderDeepAnalysisLauncher(options) {
        const cfg = options.cfg || Registry.resolvePageConfig();
        const loaded = options.loadedKeys || new Set();
        const allKeys = Registry.getPageCapabilities(Registry.currentPage(), { includeAdvanced: true });
        const advancedKeys = (options.advancedKeys || allKeys).filter((key) => !loaded.has(key));
        if (!advancedKeys.length) return null;

        const el = UI.panel("taxpayer-ai-deep-analysis-panel", "Phân tích sâu", "science", `
            <div class="flex flex-wrap gap-2">
                ${advancedKeys.slice(0, 8).map((key) => {
                    const cap = Registry.getCapability(key) || {};
                    return `
                        <button
                            type="button"
                            class="taxpayer-ai-deep-load rounded-lg border border-slate-200 bg-white px-3 py-2 text-[11px] font-black text-slate-700 transition hover:border-emerald-300 hover:bg-emerald-50"
                            data-capability="${UI.escapeHtml(key)}"
                        >
                            <span class="material-symbols-outlined mr-1 align-middle text-sm">analytics</span>
                            ${UI.escapeHtml(cap.label || key)}
                        </button>
                    `;
                }).join("")}
            </div>
            <p class="mt-3 text-[11px] text-slate-500">
                Các mô hình nặng như graph, Monte Carlo, wavelet, SVD và GraphRAG chỉ chạy khi bạn mở để tránh làm trang chính quá tải.
            </p>
        `);

        el?.querySelectorAll(".taxpayer-ai-deep-load").forEach((btn) => {
            btn.addEventListener("click", async () => {
                const key = btn.getAttribute("data-capability");
                if (!key || typeof options.loadCapability !== "function") return;
                btn.disabled = true;
                btn.classList.add("opacity-60");
                const original = btn.innerHTML;
                btn.innerHTML = `<span class="material-symbols-outlined mr-1 align-middle text-sm">progress_activity</span>Đang tải`;
                try {
                    await options.loadCapability(key, { render: true, source: "deep_analysis" });
                    btn.remove();
                } catch (error) {
                    UI.toast(error.message || "Không thể tải phân tích sâu.", "error");
                    btn.disabled = false;
                    btn.classList.remove("opacity-60");
                    btn.innerHTML = original;
                }
            });
        });
        return el;
    }

    window.TaxpayerAIPanelsAdvanced = {
        renderDeepAnalysisLauncher,
    };
})();
