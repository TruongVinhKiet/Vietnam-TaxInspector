/**
 * business_calculator.js – Frontend logic for Taxpayer Automated Tax Calculator
 * =========================================================================
 * Features:
 *   1. Integrates 8 real-time tax calculators with backend REST APIs.
 *   2. Handles session identity integration and unhides Save History triggers.
 *   3. Connects to PostgreSQL calculation history tables.
 *   4. Generates professional window.print() CSS-friendly tax receipts.
 */

let activeMst = "8092471928";
let activeName = "Cửa hàng Gia dụng Thuận Phát";
let lastCalculationResults = {};

// Helper to format currency to VND standard
function formatVND(value) {
    return new Intl.NumberFormat('vi-VN', { style: 'currency', currency: 'VND' }).format(value);
}

// Format parameter inputs into human-readable bullets
function formatInputsHTML(inputs) {
    let html = "";
    for (const [key, val] of Object.entries(inputs)) {
        let label = key;
        let formattedVal = val;
        
        if (key === "revenue" || key === "monthly_income" || key === "amount" || key === "capital" || key === "monthly_rent" || key === "contract_value" || key === "taxable_income" || key === "expenses") {
            formattedVal = formatVND(val);
        }

        const labelMap = {
            "revenue": "Doanh thu năm/tháng",
            "industry": "Ngành nghề/Lĩnh vực",
            "monthly_income": "Thu nhập bình quân tháng",
            "dependents": "Số người phụ thuộc",
            "expenses": "Chi phí thực tế",
            "amount": "Thuế gốc chậm nộp",
            "days": "Số ngày chậm nộp",
            "rate": "Tỷ lệ phạt áp dụng (%/ngày)",
            "taxable_income": "Thu nhập chịu thuế",
            "is_yearly": "Chu kỳ thu nhập năm",
            "type": "Loại hình tổ chức",
            "capital": "Vốn điều lệ ghi nhận",
            "monthly_rent": "Tiền thuê hàng tháng",
            "months": "Số tháng cho thuê",
            "contract_value": "Giá trị hợp đồng thanh toán",
            "service_type": "Loại dịch vụ nhà thầu"
        };
        
        label = labelMap[key] || key;
        html += `<p>• <strong>${label}:</strong> ${formattedVal}</p>`;
    }
    return html;
}

// Format outputs into a clean receipts block
function formatResultsHTML(results) {
    let html = "";
    for (const [key, val] of Object.entries(results)) {
        if (key === "status" || key === "industry_name" || key === "service_name" || key === "level" || key === "advice" || key === "is_taxable") continue;
        
        let label = key;
        let formattedVal = typeof val === "number" ? formatVND(val) : val;

        const labelMap = {
            "gtgt_rate_pct": "Tỷ lệ thuế GTGT (%)",
            "tncn_rate_pct": "Tỷ lệ thuế TNCN (%)",
            "tndn_rate_pct": "Tỷ lệ thuế TNDN (%)",
            "gtgt_tax": "Tiền thuế GTGT",
            "tncn_tax": "Tiền thuế TNCN",
            "tndn_tax": "Tiền thuế TNDN",
            "total_tax": "Tổng số thuế phải nộp",
            "person_deduction": "Mức giảm trừ bản thân",
            "dependent_deduction_rate": "Mức giảm trừ mỗi người phụ thuộc",
            "total_dependent_deduction": "Tổng giảm trừ người phụ thuộc",
            "total_deduction": "Tổng giảm trừ gia cảnh áp dụng",
            "taxable_income": "Thu nhập chịu thuế còn lại",
            "pit_monthly": "Thuế TNCN tạm tính theo tháng",
            "pit_total": "Tổng thuế TNCN phải nộp",
            "bracket_applied": "Bậc thuế lũy tiến cao nhất",
            "tax_khoan": "Thuế nộp khoán (%)",
            "tax_kekhai": "Thuế Kê khai (Doanh thu - Chi phí)",
            "tax_tndn_20": "Thuế TNDN 20% Doanh nghiệp",
            "tax_llc_total": "Tổng thuế nếu lên Công ty TNHH",
            "profit": "Lợi nhuận ròng hàng năm",
            "origin_amount": "Tiền thuế gốc chậm nộp",
            "days": "Số ngày chậm nộp",
            "penalty_amount": "Tiền phạt chậm nộp (0.03%/ngày)",
            "total_amount": "Tổng tiền phải nộp (cả phạt)",
            "fee": "Lệ phí môn bài phải đóng",
            "total_rent": "Tổng doanh thu thuê tài sản",
            "fee_level": "Mức lệ phí áp dụng",
            "penalty_rate": "Mức phạt áp dụng (%)"
        };

        label = labelMap[key] || key;
        html += `<p class="flex justify-between border-b border-slate-100 py-1.5">
            <span>• ${label}</span>
            <span class="font-bold text-slate-800">${formattedVal}</span>
        </p>`;
    }

    if (results.advice) {
        html += `<div class="mt-3 p-2 bg-emerald-50 border border-emerald-100 text-emerald-800 rounded font-medium">${results.advice}</div>`;
    }
    if (results.level) {
        html += `<p class="flex justify-between border-b border-slate-100 py-1.5">
            <span>• Mức phân loại áp dụng</span>
            <span class="font-bold text-emerald-600">${results.level}</span>
        </p>`;
    }
    return html;
}

// Toggle inputs for annual License Fee card (vốn for company, revenue for household)
function toggleLicenseInput() {
    const type = document.getElementById("license-type").value;
    const revBox = document.getElementById("license-rev-box");
    const capBox = document.getElementById("license-cap-box");
    
    if (type === "business") {
        revBox.classList.add("hidden");
        capBox.classList.remove("hidden");
    } else {
        revBox.classList.remove("hidden");
        capBox.classList.add("hidden");
    }
}

// Core Fetch Wrapper
async function postCalculatorAPI(endpoint, payload) {
    try {
        const response = await secureFetch(`${API_BASE}/calculator/${endpoint}`, {
            method: "POST",
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            throw new Error(`Server returned code ${response.status}`);
        }
        return await response.json();
    } catch (err) {
        console.error(`[CALC API ERR] ${endpoint}:`, err);
        alert("Lỗi khi kết nối với cổng tính thuế tự động. Hãy chắc chắn backend đã chạy.");
        return null;
    }
}

// ---- CALCULATOR HANDLERS ----

// Card 1: GTGT & TNCN Khoán
async function calculateTaxes() {
    const industry = document.getElementById("calc-industry").value;
    const revenue = parseFloat(document.getElementById("calc-revenue").value) || 0;

    const payload = { revenue, industry };
    const res = await postCalculatorAPI("gtgt-tncn", payload);

    if (res && res.status === "success") {
        lastCalculationResults["gtgt_tncn"] = { inputs: payload, results: res };
        document.getElementById("res-tax-total").textContent = formatVND(res.total_tax);
        document.getElementById("res-tax-gtgt").textContent = formatVND(res.gtgt_tax);
        document.getElementById("res-tax-tncn").textContent = formatVND(res.tncn_tax);
        
        document.getElementById("calc-tax-result").classList.remove("hidden");
    }
}

// Card 2: Deductions
async function calculateDeductions() {
    const monthly_income = parseFloat(document.getElementById("calc-monthly-income").value) || 0;
    const dependents = parseInt(document.getElementById("calc-dependents").value) || 0;

    const payload = { monthly_income, dependents };
    const res = await postCalculatorAPI("deductions", payload);

    if (res && res.status === "success") {
        lastCalculationResults["deductions"] = { inputs: payload, results: res };
        document.getElementById("res-deduct-taxable").textContent = formatVND(res.taxable_income);
        document.getElementById("res-deduct-dependents-total").textContent = formatVND(res.total_dependent_deduction);
        
        document.getElementById("calc-deduct-result").classList.remove("hidden");
    }
}

// Card 3: Comparison Matrix (Khoán vs Kê Khai vs TNHH)
async function compareMethods() {
    const revenue = parseFloat(document.getElementById("compare-rev").value) || 0;
    const expenses = parseFloat(document.getElementById("compare-exp").value) || 0;
    const industry = document.getElementById("compare-industry").value;

    const payload = { revenue, expenses, industry };
    const res = await postCalculatorAPI("compare-methods", payload);

    if (res && res.status === "success") {
        lastCalculationResults["compare"] = { inputs: payload, results: res };
        document.getElementById("res-compare-khoan").textContent = formatVND(res.tax_khoan);
        document.getElementById("res-compare-kekhai").textContent = formatVND(res.tax_kekhai);
        document.getElementById("res-compare-llc").textContent = formatVND(res.tax_llc_total);
        document.getElementById("compare-advice").textContent = res.advice;

        document.getElementById("compare-result").classList.remove("hidden");
    }
}

// Card 4: Late Penalty
async function calculateLatePenalty() {
    const amount = parseFloat(document.getElementById("late-amount").value) || 0;
    const days = parseInt(document.getElementById("late-days").value) || 0;
    const selectedRate = parseFloat(document.getElementById("late-period").value) || 0.03;

    const payload = { amount, days, rate: selectedRate };
    const res = await postCalculatorAPI("late-penalty", payload);

    if (res && res.status === "success") {
        lastCalculationResults["penalty"] = { inputs: payload, results: res };
        document.getElementById("res-late-total").textContent = formatVND(res.total_amount);
        document.getElementById("res-late-origin").textContent = formatVND(res.origin_amount);
        document.getElementById("res-late-penalty").textContent = formatVND(res.penalty_amount);

        document.getElementById("late-result").classList.remove("hidden");
    }
}

// Card 5: Progressive PIT 7 brackets
async function calculateProgressivePIT() {
    const income = parseFloat(document.getElementById("progressive-income").value) || 0;
    const cycle = document.getElementById("progressive-cycle").value;
    const is_yearly = cycle === "year";

    const payload = { taxable_income: income, is_yearly };
    const res = await postCalculatorAPI("progressive-pit", payload);

    if (res && res.status === "success") {
        lastCalculationResults["progressive"] = { inputs: payload, results: res };
        document.getElementById("res-progressive-total").textContent = formatVND(res.pit_total);
        document.getElementById("res-progressive-monthly").textContent = formatVND(res.pit_monthly);
        document.getElementById("res-progressive-bracket").textContent = `Bậc ${res.bracket_applied} (${(res.bracket_applied * 5) || 5}%)`;

        document.getElementById("progressive-result").classList.remove("hidden");
    }
}

// Card 6: License Fee
async function calculateLicenseFee() {
    const type = document.getElementById("license-type").value;
    const revenue = parseFloat(document.getElementById("license-revenue").value) || 0;
    const capital = parseFloat(document.getElementById("license-capital").value) || 0;

    const payload = { type, revenue, capital };
    const res = await postCalculatorAPI("license-fee", payload);

    if (res && res.status === "success") {
        lastCalculationResults["license_fee"] = { inputs: payload, results: res };
        document.getElementById("res-license-fee").textContent = formatVND(res.fee);
        document.getElementById("res-license-level").textContent = res.level;

        document.getElementById("license-result").classList.remove("hidden");
    }
}

// Card 7: Rental Property Tax
async function calculateRentalTax() {
    const rent = parseFloat(document.getElementById("rental-rent").value) || 0;
    const months = parseInt(document.getElementById("rental-months").value) || 12;

    const payload = { monthly_rent: rent, months };
    const res = await postCalculatorAPI("rental-tax", payload);

    if (res && res.status === "success") {
        lastCalculationResults["rental"] = { inputs: payload, results: res };
        document.getElementById("res-rental-total").textContent = formatVND(res.total_tax);
        document.getElementById("res-rental-gross").textContent = formatVND(res.total_rent);
        document.getElementById("res-rental-gtgt").textContent = formatVND(res.gtgt_tax);
        document.getElementById("res-rental-tncn").textContent = formatVND(res.tncn_tax);

        document.getElementById("rental-result").classList.remove("hidden");
    }
}

// Card 8: Contractor Tax FCT
async function calculateContractorTax() {
    const value = parseFloat(document.getElementById("contractor-val").value) || 0;
    const service = document.getElementById("contractor-service").value;

    const payload = { contract_value: value, service_type: service };
    const res = await postCalculatorAPI("contractor-tax", payload);

    if (res && res.status === "success") {
        lastCalculationResults["contractor"] = { inputs: payload, results: res };
        document.getElementById("res-contractor-total").textContent = formatVND(res.total_tax);
        document.getElementById("res-contractor-gtgt").textContent = formatVND(res.gtgt_tax);
        document.getElementById("res-contractor-tndn").textContent = formatVND(res.tndn_tax);

        document.getElementById("contractor-result").classList.remove("hidden");
    }
}


// ---- PRINT RECEIPT SERVICES (window.print Overlay) ----
function printReceipt(calcType) {
    const activeCalc = lastCalculationResults[calcType];
    if (!activeCalc) {
        alert("Vui lòng thực hiện tính toán trước khi in biên lai.");
        return;
    }

    const titleMap = {
        "gtgt_tncn": "BIÊN LAI TÍNH THUẾ GTGT & TNCN NỘP KHOÁN",
        "deductions": "BIÊN LAI TÍNH GIẢM TRỪ GIA CẢNH & THU NHẬP CHỊU THUẾ",
        "compare": "BẢN PHÂN TÍCH TỐI ƯU HÓA PHƯƠNG PHÁP THUẾ & MÔ HÌNH",
        "penalty": "BIÊN LAI TÍNH PHẠT CHẬM NỘP THUẾ",
        "progressive": "BIÊN LAI TÍNH THUẾ TNCN LŨY TIẾN 7 BẬC",
        "license_fee": "BIÊN LAI XÁC ĐỊNH LỆ PHÍ MÔN BÀI HÀNG NĂM",
        "rental": "BIÊN LAI TÍNH THUẾ CHO THUÊ TÀI SẢN / BẤT ĐỘNG SẢN",
        "contractor": "BIÊN LAI TÍNH THUẾ NHÀ THẦU NƯỚC NGOÀI (FCT)"
    };

    // Populate Print Area
    document.getElementById("print-title").textContent = titleMap[calcType] || "BIÊN LAI TÍNH TOÁN THUẾ TỰ ĐỘNG";
    document.getElementById("print-date").textContent = `Thời gian xuất: ${new Date().toLocaleString('vi-VN')}`;
    document.getElementById("print-user-name").textContent = activeName;
    document.getElementById("print-user-mst").textContent = activeMst;

    document.getElementById("print-inputs").innerHTML = formatInputsHTML(activeCalc.inputs);
    document.getElementById("print-results").innerHTML = formatResultsHTML(activeCalc.results);

    // Call window print
    window.print();
}


// ---- HISTORY LOGGING SERVICES ----

// Save to DB History
async function saveToHistory(calcType) {
    const activeCalc = lastCalculationResults[calcType];
    if (!activeCalc) {
        alert("Vui lòng thực hiện tính toán trước khi lưu.");
        return;
    }

    try {
        const payload = {
            tax_code: activeMst,
            calc_type: calcType,
            inputs: activeCalc.inputs,
            results: activeCalc.results
        };

        const res = await secureFetch(`${API_BASE}/calculator/history`, {
            method: "POST",
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            alert("Lưu lịch sử tính toán thuế thành công!");
            loadHistory();
        } else {
            alert("Không thể lưu lịch sử. Vui lòng kiểm tra đăng nhập.");
        }
    } catch (err) {
        console.error("Save history error:", err);
        alert("Đã xảy ra lỗi khi kết nối lưu lịch sử.");
    }
}

// Load DB History
async function loadHistory() {
    if (!activeMst) return;

    try {
        const response = await secureFetch(`${API_BASE}/calculator/history/${activeMst}`);
        if (!response.ok) return;

        const data = await response.json();
        const historyList = data.history || [];

        document.getElementById("history-count").textContent = `${historyList.length} kết quả`;

        const tbody = document.getElementById("history-table-body");
        if (historyList.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" class="py-8 text-center text-slate-400 italic">Chưa thực hiện phép tính nào hoặc chưa đăng nhập để đồng bộ lịch sử.</td></tr>`;
            return;
        }

        let html = "";
        historyList.forEach(item => {
            const inputsFormatted = formatInputsHTML(item.inputs);
            
            // Extract the main result field
            let finalTax = 0;
            if (item.results.total_tax !== undefined) finalTax = item.results.total_tax;
            else if (item.results.pit_total !== undefined) finalTax = item.results.pit_total;
            else if (item.results.taxable_income !== undefined) finalTax = item.results.taxable_income;
            else if (item.results.total_amount !== undefined) finalTax = item.results.total_amount;
            else if (item.results.fee !== undefined) finalTax = item.results.fee;

            const finalTaxVND = formatVND(finalTax);

            html += `<tr class="border-b border-slate-100 hover:bg-slate-50/50 transition-colors">
                <td class="py-3 font-mono text-[10px] text-slate-400">${item.created_at}</td>
                <td class="py-3 font-semibold text-slate-800">${item.calc_name}</td>
                <td class="py-3 text-[10px] text-slate-500 leading-relaxed">${inputsFormatted}</td>
                <td class="py-3 text-right font-bold text-rose-600">${finalTaxVND}</td>
                <td class="py-3 text-center">
                    <button onclick='recallAndPrint(${JSON.stringify(item)})' class="text-emerald-600 hover:text-emerald-700 font-bold flex items-center gap-0.5 mx-auto text-[10px]">
                        <span class="material-symbols-outlined text-sm">print</span> In
                    </button>
                </td>
            </tr>`;
        });

        tbody.innerHTML = html;

    } catch (err) {
        console.error("Load history error:", err);
    }
}

// Recall from history row and print
function recallAndPrint(historyItem) {
    lastCalculationResults[historyItem.calc_type] = {
        inputs: historyItem.inputs,
        results: historyItem.results
    };
    printReceipt(historyItem.calc_type);
}


// ---- AUTHENTICATION AND SIDEBAR INITS ----

document.addEventListener("DOMContentLoaded", async () => {
    // 1. Enforce active sidebar highlighted item
    const currentPage = window.location.pathname.split("/").pop() || 'business_dashboard.html';
    const navItems = document.querySelectorAll('#sidebar-nav .nav-item');
    let pageTitle = "Hệ thống";

    navItems.forEach(item => {
        if (item.getAttribute('data-page') === currentPage) {
            item.classList.remove('text-slate-400', 'hover:bg-white/5');
            item.classList.add('bg-emerald-500/10', 'text-emerald-400', 'font-semibold');
            pageTitle = item.getAttribute('data-title');
        }
    });

    const breadcrumbCurrent = document.getElementById('breadcrumb-current');
    if (breadcrumbCurrent) breadcrumbCurrent.textContent = pageTitle;
    const sidebarSubtitle = document.getElementById('sidebar-subtitle');
    if (sidebarSubtitle) sidebarSubtitle.textContent = pageTitle;

    // 2. Load taxpayer session info
    if (typeof hydrateSidebarIdentity === "function") {
        try {
            const user = await hydrateSidebarIdentity();
            if (user) {
                activeMst = user.badge_id || "8092471928";
                activeName = user.full_name || "Cửa hàng Gia dụng Thuận Phát";

                // Setup header bindings manually for safety
                const headerFullName = document.getElementById("header-user-fullname");
                const headerMst = document.getElementById("header-user-mst");
                const headerFallback = document.getElementById("header-avatar-fallback");

                if (headerFullName) headerFullName.textContent = activeName;
                if (headerMst) headerMst.textContent = `MST: ${activeMst}`;
                if (headerFallback) {
                    headerFallback.textContent = activeName
                        .split(" ")
                        .filter(Boolean)
                        .slice(0, 2)
                        .map(w => w[0])
                        .join("")
                        .toUpperCase();
                }

                // Show save history buttons across all forms
                document.querySelectorAll("[id^='save-']").forEach(btn => {
                    btn.classList.remove("hidden");
                });

                // Load database history logs
                loadHistory();
            }
        } catch (e) {
            console.warn("[CALC INITS] Hydrate taxpayer info failed:", e);
        }
    }
});
