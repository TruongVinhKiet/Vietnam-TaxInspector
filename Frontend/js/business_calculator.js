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
        
        if (key === "revenue" || key === "monthly_income" || key === "amount" || key === "capital" || key === "monthly_rent" || key === "contract_value" || key === "taxable_income" || key === "expenses" || key === "revenue_mean" || key === "current_price" || key === "new_price" || key === "target_unit_price") {
            formattedVal = formatVND(val);
        }

        const labelMap = {
            "revenue": "Doanh thu năm/tháng",
            "revenue_mean": "Doanh thu trung bình dự kiến",
            "volatility_pct": "Độ biến động doanh thu (%)",
            "expense_ratio_pct": "Tỷ lệ chi phí thực tế (%)",
            "tax_rate_pct": "Thuế suất áp dụng (%)",
            "iterations": "Số kịch bản mô phỏng",
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
            "service_type": "Loại dịch vụ nhà thầu",
            "current_price": "Giá bán hiện tại",
            "current_quantity": "Sản lượng bán hiện tại",
            "new_price": "Giá bán mới đề xuất",
            "elasticity_coefficient": "Hệ số co giãn (PED)",
            "target_unit_price": "Giá đơn vị liên kết đề xuất",
            "target_quantity": "Sản lượng liên kết đề xuất"
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
        if (key === "status" || key === "industry_name" || key === "service_name" || key === "level" || key === "advice" || key === "is_taxable" || key === "bins" || key === "verdict_color") continue;
        
        let label = key;
        if (typeof val === "object" && val !== null) {
            for (const [subKey, subVal] of Object.entries(val)) {
                let subLabel = `${key} - ${subKey}`;
                if (key === "percentiles") {
                    subLabel = `Phân vị ${subKey} (Thuế)`;
                } else if (key === "value_at_risk_95") {
                    subLabel = `VaR 95% (${subKey === 'tax' ? 'Thuế' : 'Doanh thu'})`;
                } else if (key === "explanation" && subKey === "optimal_pricing") {
                    subLabel = "Giá bán tối ưu hóa doanh thu đề xuất";
                } else if (key === "arms_length_range") {
                    subLabel = `Biên Arm's Length (${subKey === 'min' ? 'Tối thiểu' : 'Tối đa'})`;
                } else if (key === "gev_parameters") {
                    subLabel = `Tham số GEV Gumbel (${subKey})`;
                } else if (key === "return_levels") {
                    subLabel = `Chu kỳ lặp stress ${subKey}`;
                }
                html += `<p class="flex justify-between border-b border-slate-100 py-1.5 text-[11px]">
                    <span>• ${subLabel}</span>
                    <span class="font-bold text-slate-800">${typeof subVal === "number" ? formatVND(subVal) : subVal}</span>
                </p>`;
            }
            continue;
        }
        
        // Don't format rates, percentages, verdicts or quantities as raw VND
        let formattedVal = (typeof val === "number" && 
            !["revenue_change_pct", "current_quantity", "new_quantity", "days", "months", "mahalanobis_distance", "p_value", "extreme_stress_probability"].includes(key)) 
            ? formatVND(val) 
            : val;

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
            "penalty_rate": "Mức phạt áp dụng (%)",
            "current_revenue": "Doanh thu hiện tại",
            "new_revenue": "Doanh thu mới giả lập",
            "revenue_change": "Biến động Doanh thu",
            "revenue_change_pct": "Biến động Doanh thu (%)",
            "current_quantity": "Sản lượng hiện tại",
            "new_quantity": "Sản lượng mới giả lập",
            "gtgt_tax_change": "Biến động thuế GTGT",
            "tncn_tax_change": "Biến động thuế TNCN",
            "total_tax_change": "Biến động Tổng thuế",
            "verdict_label": "Phân loại nhu cầu thị trường",
            "mahalanobis_distance": "Khoảng cách Mahalanobis",
            "p_value": "Chỉ số tin cậy (p-value)",
            "risk_level": "Đánh giá mức độ rủi ro chuyển giá",
            "verdict": "Kết luận kiểm toán Nghị định 132",
            "value_at_risk_99": "Rủi ro thanh khoản thuế VaR 99%",
            "expected_shortfall_99": "Tổn thất thâm hụt dự kiến ES 99%",
            "extreme_stress_probability": "Tần suất xảy ra căng thẳng thanh khoản"
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

        // Calculate ratios relative to annual revenue
        const pctKhoan = revenue > 0 ? (res.tax_khoan / revenue * 100) : 0;
        const pctKekhai = revenue > 0 ? (res.tax_kekhai / revenue * 100) : 0;
        const pctLlc = revenue > 0 ? (res.tax_llc_total / revenue * 100) : 0;

        document.getElementById("pct-compare-khoan").textContent = `${pctKhoan.toFixed(1)}%`;
        document.getElementById("pct-compare-kekhai").textContent = `${pctKekhai.toFixed(1)}%`;
        document.getElementById("pct-compare-llc").textContent = `${pctLlc.toFixed(1)}%`;

        // Normalize bar widths (relative scale relative to the highest tax pct)
        const maxPct = Math.max(pctKhoan, pctKekhai, pctLlc, 1.0);
        document.getElementById("bar-compare-khoan").style.width = `${(pctKhoan / maxPct * 100).toFixed(0)}%`;
        document.getElementById("bar-compare-kekhai").style.width = `${(pctKekhai / maxPct * 100).toFixed(0)}%`;
        document.getElementById("bar-compare-llc").style.width = `${(pctLlc / maxPct * 100).toFixed(0)}%`;

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


// Card 9: Mô phỏng Monte Carlo (AI-Powered)
async function runMonteCarloSimulation() {
    const revenueMean = parseFloat(document.getElementById("monte-revenue").value) || 0;
    const volatility = parseFloat(document.getElementById("monte-volatility").value) || 20;
    const taxRate = parseFloat(document.getElementById("monte-tax-rate").value) || 1.5;
    const iterations = parseInt(document.getElementById("monte-iterations").value) || 10000;

    if (revenueMean <= 0) {
        alert("Vui lòng nhập doanh thu dự tính lớn hơn 0.");
        return;
    }

    const payload = {
        revenue_mean: revenueMean,
        volatility_pct: volatility,
        expense_ratio_pct: 50.0, // estimate default expense ratio
        tax_rate_pct: taxRate,
        iterations: iterations
    };

    // Trigger API calling
    try {
        const response = await secureFetch(`${API_BASE}/taxpayer/intelligence/monte-carlo-simulation`, {
            method: "POST",
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            throw new Error(`Server returned code ${response.status}`);
        }
        const res = await response.json();

        if (res && res.status === "success") {
            lastCalculationResults["monte_carlo"] = { inputs: payload, results: res };
            
            // Populate results HTML
            document.getElementById("res-monte-var").textContent = formatVND(res.value_at_risk_95.tax);
            document.getElementById("res-monte-p5").textContent = formatVND(res.percentiles.P5.tax);
            document.getElementById("res-monte-p25").textContent = formatVND(res.percentiles.P25.tax);
            document.getElementById("res-monte-p50").textContent = formatVND(res.percentiles.P50.tax);
            document.getElementById("res-monte-p75").textContent = formatVND(res.percentiles.P75.tax);
            document.getElementById("res-monte-p95").textContent = formatVND(res.percentiles.P95.tax);
            document.getElementById("res-monte-message").textContent = res.risk_message;

            // Render histogram representation
            const bins = res.bins || [];
            const maxFreq = Math.max(...bins.map(b => b.frequency || 1));
            const histContainer = document.getElementById("monte-histogram");
            
            let histHtml = "";
            bins.forEach(bin => {
                const heightPct = (bin.frequency / maxFreq) * 100;
                // color gradient based on tax level (redder on the right)
                const isVaR = bin.bin_index >= 8;
                const barColor = isVaR ? "bg-rose-500 hover:bg-rose-600" : "bg-indigo-400 hover:bg-indigo-500";
                histHtml += `
                    <div class="flex-1 h-full flex flex-col justify-end group relative" title="Khoảng: ${formatVND(bin.range_start)} - ${formatVND(bin.range_end)}\nTần suất: ${bin.frequency} kịch bản (${bin.pct}%)">
                        <div class="${barColor} rounded-t transition-all" style="height: ${Math.max(4, heightPct)}%"></div>
                    </div>
                `;
            });
            histContainer.innerHTML = histHtml;

            // Reveal result
            document.getElementById("monte-result").classList.remove("hidden");
        }
    } catch (err) {
        console.error("[MONTE CARLO ERROR]", err);
        alert("Lỗi khi kết nối với máy chủ AI Monte Carlo.");
    }
}


// Card 10: Phân tích Điểm hòa vốn & Lợi nhuận mục tiêu CVP (AI-Powered)
async function runBreakevenAnalysis() {
    const fixedCosts = parseFloat(document.getElementById("cvp-fixed-costs").value) || 0;
    const variableRatio = parseFloat(document.getElementById("cvp-variable-ratio").value) || 0;
    const currentRevenue = parseFloat(document.getElementById("cvp-current-revenue").value) || 0;
    const targetProfit = parseFloat(document.getElementById("cvp-target-profit").value) || 0;

    if (fixedCosts <= 0) {
        alert("Vui lòng nhập chi phí cố định lớn hơn 0.");
        return;
    }

    const payload = {
        fixed_costs: fixedCosts,
        variable_cost_ratio_pct: variableRatio,
        current_revenue: currentRevenue,
        target_profit: targetProfit
    };

    try {
        const response = await secureFetch(`${API_BASE}/taxpayer/intelligence/breakeven-analysis`, {
            method: "POST",
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            throw new Error(`Server returned code ${response.status}`);
        }
        const res = await response.json();

        if (res && res.status === "success") {
            lastCalculationResults["breakeven"] = { inputs: payload, results: res };
            
            // Populate results
            document.getElementById("res-cvp-breakeven").textContent = formatVND(res.breakeven_revenue);
            document.getElementById("res-cvp-safety-margin").textContent = `${res.safety_margin_pct}%`;
            document.getElementById("res-cvp-target-revenue").textContent = formatVND(res.target_revenue);
            
            const verdictEl = document.getElementById("res-cvp-verdict-label");
            verdictEl.textContent = res.verdict_label;
            verdictEl.className = `font-bold text-${res.verdict_color}-600`;

            // Bar color & width
            const barEl = document.getElementById("res-cvp-bar");
            const pctWidth = Math.min(100, Math.max(0, (currentRevenue / (res.breakeven_revenue || 1)) * 100));
            barEl.style.width = `${pctWidth}%`;
            barEl.className = `h-full transition-all bg-${res.verdict_color}-500`;

            // Counterfactual / reduce fixed advice
            document.getElementById("res-cvp-message").textContent = 
                `${res.explanation.methodology} Khuyên dùng: ${res.explanation.counterfactual.reduce_fixed}`;

            // Reveal save button
            document.getElementById("save-cvp-btn").classList.remove("hidden");

            // Reveal result
            document.getElementById("cvp-result").classList.remove("hidden");
        }
    } catch (err) {
        console.error("[CVP ERROR]", err);
        alert("Lỗi khi kết nối với máy chủ AI CVP.");
    }
}

async function runPriceElasticitySimulation() {
    const currentPrice = parseFloat(document.getElementById("elasticity-current-price").value) || 0;
    const currentQuantity = parseFloat(document.getElementById("elasticity-current-qty").value) || 0;
    const newPrice = parseFloat(document.getElementById("elasticity-new-price").value) || 0;
    const elasticityCoeff = parseFloat(document.getElementById("elasticity-coeff").value) || -1.5;

    if (currentPrice <= 0 || currentQuantity <= 0 || newPrice <= 0) {
        alert("Vui lòng nhập giá hiện tại, sản lượng hiện tại và giá bán mới lớn hơn 0.");
        return;
    }

    const payload = {
        current_price: currentPrice,
        current_quantity: currentQuantity,
        new_price: newPrice,
        elasticity_coefficient: elasticityCoeff
    };

    try {
        const response = await secureFetch(`${API_BASE}/taxpayer/intelligence/price-elasticity`, {
            method: "POST",
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            throw new Error(`Server returned code ${response.status}`);
        }
        const res = await response.json();

        if (res && res.status === "success") {
            lastCalculationResults["price_elasticity"] = { inputs: payload, results: res };
            
            // Populate results
            document.getElementById("res-elasticity-cur-rev").textContent = formatVND(res.current_revenue);
            document.getElementById("res-elasticity-new-rev").textContent = formatVND(res.new_revenue);
            
            const changeEl = document.getElementById("res-elasticity-change-rev");
            changeEl.textContent = `${res.revenue_change >= 0 ? '+' : ''}${formatVND(res.revenue_change)} (${res.revenue_change_pct}%)`;
            changeEl.className = res.revenue_change >= 0 ? "font-bold text-emerald-600" : "font-bold text-rose-600";
            
            document.getElementById("res-elasticity-new-qty").textContent = `${Math.round(res.new_quantity)} sản phẩm`;
            
            const taxEl = document.getElementById("res-elasticity-tax-change");
            taxEl.textContent = `${res.total_tax_change >= 0 ? '+' : ''}${formatVND(res.total_tax_change)}`;
            taxEl.className = res.total_tax_change >= 0 ? "font-bold text-emerald-600" : "font-bold text-rose-600";
            
            const verdictEl = document.getElementById("res-elasticity-verdict");
            verdictEl.textContent = res.verdict_label;
            verdictEl.className = `font-bold text-${res.verdict_color}-600`;
            
            document.getElementById("res-elasticity-message").textContent = 
                `${res.explanation.methodology} Khuyên dùng: ${res.advice}`;

            // Reveal save button
            document.getElementById("save-elasticity-btn").classList.remove("hidden");

            // Reveal result
            document.getElementById("elasticity-result").classList.remove("hidden");
        }
    } catch (err) {
        console.error("[ELASTICITY ERROR]", err);
        alert("Lỗi khi kết nối với máy chủ AI Độ nhạy doanh thu.");
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
        "contractor": "BIÊN LAI TÍNH THUẾ NHÀ THẦU NƯỚC NGOÀI (FCT)",
        "monte_carlo": "AI FORECAST - BẢN MÔ PHỎNG RỦI RO THUẾ MONTE CARLO (10K RUNS)",
        "breakeven": "AI ANALYTICS - BẢN PHÂN TÍCH ĐIỂM HÒA VỐN & CHI PHÍ CVP",
        "price_elasticity": "BẢN PHÂN TÍCH ĐỘ NHẠY GIÁ & DOANH THU (PED)",
        "transfer_pricing": "AI TP DIAGNOSTIC - BẢN ĐÁNH GIÁ RỦI RO GIAO DỊCH LIÊN KẾT (NĐ 132)",
        "outflow_stress": "AI EVT STRESS SIMULATOR - BẢN MÔ PHỎNG KHỦNG HOẢNG NỢ THUẾ GEV"
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


// ---- F19 & F20 FLAGSHIP INTELLIGENCE FEATURES ----

// Card 12: Giao dịch liên kết Transfer Pricing (F19)
async function runTransferPricingEvaluation() {
    const targetPrice = parseFloat(document.getElementById("tp-unit-price").value) || 0;
    const targetQty = parseFloat(document.getElementById("tp-qty").value) || 0;

    if (targetPrice <= 0 || targetQty <= 0) {
        alert("Vui lòng nhập giá đơn vị và sản lượng đề xuất lớn hơn 0.");
        return;
    }

    const payload = {
        target_unit_price: targetPrice,
        target_quantity: targetQty
    };

    try {
        const response = await secureFetch(`${API_BASE}/taxpayer/intelligence/transfer-pricing`, {
            method: "POST",
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            throw new Error(`Server returned code ${response.status}`);
        }
        const res = await response.json();

        if (res && res.status === "success") {
            lastCalculationResults["transfer_pricing"] = { inputs: payload, results: res };

            // Populate results
            const riskEl = document.getElementById("res-tp-risk");
            if (riskEl) {
                riskEl.textContent = res.risk_level === 'high' ? 'Nguy cơ cao' : res.risk_level === 'medium' ? 'Trung bình' : 'An toàn';
                riskEl.className = res.risk_level === 'high' ? "font-black text-rose-600 text-sm uppercase" : res.risk_level === 'medium' ? "font-black text-amber-600 text-sm uppercase" : "font-black text-emerald-600 text-sm uppercase";
            }

            const distEl = document.getElementById("res-tp-dist");
            if (distEl) distEl.textContent = res.mahalanobis_distance;

            const pvalEl = document.getElementById("res-tp-pvalue");
            if (pvalEl) pvalEl.textContent = res.p_value;

            const rangeEl = document.getElementById("res-tp-range");
            if (rangeEl) rangeEl.textContent = `${formatVND(res.arms_length_range.min)} - ${formatVND(res.arms_length_range.max)}`;

            const verdictEl = document.getElementById("res-tp-verdict");
            if (verdictEl) verdictEl.textContent = res.verdict;

            const adviceEl = document.getElementById("res-tp-advice");
            if (adviceEl) adviceEl.textContent = `Phương án đề xuất: ${res.explanation.counterfactual.adjust_price}`;

            // Reveal save button
            const saveBtn = document.getElementById("save-tp-btn");
            if (saveBtn) saveBtn.classList.remove("hidden");

            // Reveal result
            const resultDiv = document.getElementById("tp-result");
            if (resultDiv) resultDiv.classList.remove("hidden");
        }
    } catch (err) {
        console.error("[TRANSFER PRICING ERROR]", err);
        alert("Lỗi khi kết nối với máy chủ AI Giao dịch liên kết.");
    }
}

// Card 13: GEV Outflow Stress Simulation (F20)
async function runTaxOutflowStressSimulation() {
    try {
        const response = await secureFetch(`${API_BASE}/taxpayer/intelligence/outflow-stress`, {
            method: "POST",
            body: JSON.stringify({})
        });
        if (!response.ok) {
            throw new Error(`Server returned code ${response.status}`);
        }
        const res = await response.json();

        if (res && res.status === "success") {
            lastCalculationResults["outflow_stress"] = { inputs: {}, results: res };

            // Populate results
            const varEl = document.getElementById("res-stress-var");
            if (varEl) varEl.textContent = formatVND(res.value_at_risk_99);

            const esEl = document.getElementById("res-stress-es");
            if (esEl) esEl.textContent = formatVND(res.expected_shortfall_99);

            const probEl = document.getElementById("res-stress-prob");
            if (probEl) probEl.textContent = `${(res.extreme_stress_probability * 100).toFixed(1)}%`;

            const t12El = document.getElementById("res-stress-t12");
            if (t12El) t12El.textContent = formatVND(res.return_levels.T_12_months);

            const t24El = document.getElementById("res-stress-t24");
            if (t24El) t24El.textContent = formatVND(res.return_levels.T_24_months);

            const t60El = document.getElementById("res-stress-t60");
            if (t60El) t60El.textContent = formatVND(res.return_levels.T_60_months);

            const t100El = document.getElementById("res-stress-t100");
            if (t100El) t100El.textContent = formatVND(res.return_levels.T_100_months);

            const verdictEl = document.getElementById("res-stress-verdict");
            if (verdictEl) verdictEl.textContent = `Dự phòng an toàn đề xuất: ${res.explanation.counterfactual.reserve_cash}`;

            // Reveal save button
            const saveBtn = document.getElementById("save-stress-btn");
            if (saveBtn) saveBtn.classList.remove("hidden");

            // Reveal result
            const resultDiv = document.getElementById("stress-result");
            if (resultDiv) resultDiv.classList.remove("hidden");
        }
    } catch (err) {
        console.error("[OUTFLOW STRESS ERROR]", err);
        alert("Lỗi khi kết nối với máy chủ AI Stress Simulation.");
    }
}

// Bind to window context
window.runTransferPricingEvaluation = runTransferPricingEvaluation;
window.runTaxOutflowStressSimulation = runTaxOutflowStressSimulation;


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
