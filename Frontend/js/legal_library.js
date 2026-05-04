let LEGAL_DOCS = [];

function getStampSVG(agency) {
    const textPathStr = agency === "QUỐC HỘI" ? "QUỐC HỘI NƯỚC CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" :
                        agency === "CHÍNH PHỦ" ? "CHÍNH PHỦ NƯỚC CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" :
                        agency === "BỘ TÀI CHÍNH" ? "BỘ TÀI CHÍNH - CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" : "TỔNG CỤC THUẾ - BỘ TÀI CHÍNH";
    const innerText = agency === "QUỐC HỘI" ? "QUỐC HỘI" : 
                      agency === "CHÍNH PHỦ" ? "CHÍNH PHỦ" :
                      agency === "BỘ TÀI CHÍNH" ? "BỘ TÀI CHÍNH" : "TỔNG CỤC THUẾ";
    
    return `
        <svg width="140" height="140" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <circle cx="100" cy="100" r="95" fill="none" stroke="#dc2626" stroke-width="4"/>
            <circle cx="100" cy="100" r="88" fill="none" stroke="#dc2626" stroke-width="2"/>
            <circle cx="100" cy="100" r="65" fill="none" stroke="#dc2626" stroke-width="2"/>
            
            <path id="stamp-text-path" d="M 25,100 A 75,75 0 1,1 175,100 A 75,75 0 1,1 25,100" fill="none" />
            <text fill="#dc2626" font-family="Arial" font-weight="bold" font-size="14" letter-spacing="2">
                <textPath href="#stamp-text-path" startOffset="50%" text-anchor="middle">
                    ${textPathStr}
                </textPath>
            </text>
            
            <!-- Star in center -->
            <polygon points="100,55 110,85 140,85 115,105 125,135 100,115 75,135 85,105 60,85 90,85" fill="#dc2626"/>
        </svg>
    `;
}

document.addEventListener("DOMContentLoaded", () => {
    const grid = document.getElementById("legal-documents-grid");
    const modal = document.getElementById("document-modal");
    const backdrop = document.getElementById("document-modal-backdrop");
    const closeBtn = document.getElementById("close-document-btn");
    const filter = document.getElementById("doc-type-filter");

    // Fetch Data from DB
    async function loadDocuments(docType) {
        try {
            let url = 'http://localhost:8000/api/legal/documents';
            if (docType) url += '?doc_type=' + encodeURIComponent(docType);
            const response = await fetch(url);
            const data = await response.json();
            LEGAL_DOCS = data.documents;
            
            renderDocuments(LEGAL_DOCS);
        } catch (error) {
            console.error("Error loading legal documents:", error);
            grid.innerHTML = `<div class="text-center py-10 col-span-full text-red-500">
                Lỗi tải danh sách văn bản pháp luật.
            </div>`;
        }
    }

    function renderDocuments(docs) {
        grid.innerHTML = "";
        if (!docs || docs.length === 0) {
                grid.innerHTML = `<div class="text-center py-10 col-span-full">
                    <p class="text-slate-500 text-sm">Chưa có văn bản pháp luật nào.</p>
                </div>`;
                return;
            }

            LEGAL_DOCS.forEach((doc, index) => {
                const card = document.createElement("div");
                // Add staggered fade-in animation
                card.className = "bg-white rounded-xl p-6 shadow-sm border border-slate-200 hover:shadow-xl hover:-translate-y-1 hover:border-primary-container transition-all duration-300 cursor-pointer group flex flex-col h-full opacity-0 translate-y-4";
                card.style.animation = `fadeInUp 0.5s ease forwards ${index * 0.1}s`;
                
                card.innerHTML = `
                    <div class="flex justify-between items-start mb-4">
                        <span class="px-2.5 py-1 text-[10px] font-bold uppercase tracking-widest rounded bg-primary-container text-white">
                            ${doc.type || 'VĂN BẢN'}
                        </span>
                        <span class="text-xs font-bold text-slate-400 group-hover:text-primary-container transition-colors">${doc.number || 'Chưa rõ số'}</span>
                    </div>
                    <h3 class="text-lg font-bold text-slate-800 mb-2 group-hover:text-primary-container transition-colors line-clamp-2">${doc.title || 'Văn bản'}</h3>
                    <p class="text-xs text-slate-500 mb-6 flex-1">${doc.agency || 'Cơ quan'} ${doc.subAgency ? ' - ' + doc.subAgency : ''}</p>
                    <div class="flex flex-wrap gap-2 mt-auto">
                        ${(doc.tags || []).map(tag => `<span class="px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-[10px] font-medium">${tag}</span>`).join('')}
                    </div>
                `;
                
                card.addEventListener("click", () => openDocument(doc));
                grid.appendChild(card);
            });
    }

    loadDocuments();

    if (filter) {
        filter.addEventListener("change", (e) => {
            loadDocuments(e.target.value);
            if (searchInput) searchInput.value = ""; // Reset search when changing filter
        });
    }

    const searchInput = document.getElementById("doc-search-input");
    const searchBtn = document.getElementById("doc-search-btn");

    function handleSearch() {
        const query = (searchInput.value || "").toLowerCase().trim();
        if (!query) {
            renderDocuments(LEGAL_DOCS);
            return;
        }
        
        const filteredDocs = LEGAL_DOCS.filter(doc => {
            const titleMatch = (doc.title || "").toLowerCase().includes(query);
            const numberMatch = (doc.number || "").toLowerCase().includes(query);
            const contentMatch = (doc.content || "").toLowerCase().includes(query);
            return titleMatch || numberMatch || contentMatch;
        });
        
        renderDocuments(filteredDocs);
    }

    if (searchBtn) searchBtn.addEventListener("click", handleSearch);
    if (searchInput) {
        searchInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") handleSearch();
        });
    }

    // Dynamic keyframes for fadeInUp
    const style = document.createElement('style');
    style.innerHTML = `
        @keyframes fadeInUp {
            to { opacity: 1; transform: translateY(0); }
        }
    `;
    document.head.appendChild(style);

    // Modal Logic
    function openDocument(doc) {
        document.getElementById("doc-agency").textContent = doc.agency || "";
        document.getElementById("doc-sub-agency").textContent = doc.subAgency || "";
        document.getElementById("doc-number").textContent = doc.number || "";
        document.querySelector("#document-modal .italic.text-right").textContent = doc.date || "";
        document.getElementById("doc-type-title").textContent = doc.type || "";
        document.getElementById("doc-main-title").textContent = doc.title || "";
        document.getElementById("doc-content").textContent = doc.content || "";
        document.getElementById("doc-signer-title").innerHTML = (doc.signerTitle || "").replace(/\n/g, '<br/>');
        document.getElementById("doc-signer-name").textContent = doc.signerName || "";
        
        document.getElementById("doc-stamp-container").innerHTML = getStampSVG(doc.agency || "");

        // Render GraphRAG v2 Metadata
        const metaContainer = document.getElementById("doc-metadata-container");
        if (metaContainer) {
            let metaHtml = "";
            if (doc.effective_status) {
                const isExpired = doc.effective_status.toLowerCase().includes("hết hiệu lực");
                metaHtml += `<span class="px-3 py-1 ${isExpired ? 'bg-red-100 text-red-700' : 'bg-emerald-100 text-emerald-700'} rounded font-semibold border ${isExpired ? 'border-red-200' : 'border-emerald-200'} shadow-sm"><i class="fa-solid fa-file-signature mr-1"></i> ${doc.effective_status}</span>`;
            }
            if (doc.official_letter_scope) {
                metaHtml += `<span class="px-3 py-1 bg-sky-100 text-sky-700 rounded font-semibold border border-sky-200 shadow-sm"><i class="fa-solid fa-map-location-dot mr-1"></i> Phạm vi: ${doc.official_letter_scope}</span>`;
            }
            if (doc.authority_path) {
                metaHtml += `<span class="px-3 py-1 bg-slate-100 text-slate-700 rounded font-semibold border border-slate-200 shadow-sm"><i class="fa-solid fa-sitemap mr-1"></i> ${doc.authority_path}</span>`;
            }
            metaContainer.innerHTML = metaHtml || `<span class="px-3 py-1 bg-slate-100 text-slate-500 rounded font-medium">Không có siêu dữ liệu GraphRAG</span>`;
        }

        modal.classList.remove("hidden");
        document.body.style.overflow = "hidden"; // Prevent background scrolling
        
        // Reset scroll position to top
        const scrollContainer = modal.querySelector('.overflow-y-auto');
        if (scrollContainer) scrollContainer.scrollTop = 0;
    }

    function closeDocument() {
        modal.classList.add("hidden");
        document.body.style.overflow = "";
    }

    closeBtn.addEventListener("click", closeDocument);
    backdrop.addEventListener("click", closeDocument);
});
